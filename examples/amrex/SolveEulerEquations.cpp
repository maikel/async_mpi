#include <ampi/amrex/FillBoundary.hpp>

#include <ampi/bulk_sequence.hpp>
#include <ampi/for_each.hpp>
#include <ampi/bulk_on.hpp>
#include <ampi/tbb_task_scheduler.hpp>

#include <ampi/amrex/ReduceMax.hpp>

#include <unifex/single_thread_context.hpp>
#include <unifex/bulk_join.hpp>
#include <unifex/bulk_transform.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/static_thread_pool.hpp>

#include <ranges>

#include "AMReX_NonLocalBC.H"

#include "AMReX.H"
#include "AMReX_AmrCore.H"
#include "AMReX_MultiFab.H"

using namespace amrex;

enum Dims { ix, iy, iz };

static constexpr IntVect e_x = IntVect::TheDimensionVector(ix);
static constexpr IntVect e_y = IntVect::TheDimensionVector(iy);
static constexpr IntVect e_z = IntVect::TheDimensionVector(iz);

struct EulerEquation {
  explicit EulerEquation(Real gamma_) : gamma{gamma_} {}

  Real gamma{1.4};
  Real gamma_inv{1.0 / gamma};
  Real gm1{(gamma - 1.0)};
  Real gm1_inv{1.0 / (gamma - 1.0)};

  AMREX_GPU_HOST_DEVICE static Real
  KineticEnergyDensity(Real rho, Real rhou, Real rhov, Real rhow) noexcept {
    const Real rhoE_kin = 0.5 * (rhou * rhou + rhov * rhov + rhow * rhow) / rho;
    return rhoE_kin;
  }

  AMREX_GPU_HOST_DEVICE Real
  Pressure(Real rho, Real rhou, Real rhov, Real rhow, Real rhoE) const noexcept {
    const Real rhoE_kin = KineticEnergyDensity(rho, rhou, rhov, rhow);
    const Real p = (rhoE - rhoE_kin) * gm1_inv;
    AMREX_ASSERT(p > 0.0);
    return p;
  }
};

Dim3 Shift(Dim3 i, Direction dir, int n) noexcept {
  int array[3]{i.x, i.y, i.z};
  std::size_t dir_s = static_cast<std::size_t>(dir);
  AMREX_ASSERT(dir_s < 3);
  array[dir_s] += n;
  return Dim3{array[0], array[1], array[2]};
}

class EulerAmrCore : public AmrCore {
  enum { coarsest_level };
  enum { RHO, RHOU, RHOV, RHOW, RHOE, n_components };

  tbb::task_arena arena{};
  unifex::static_thread_pool communication_thread_pool{2};

  MultiFab states;
  std::array<MultiFab, AMREX_SPACEDIM> fluxes;

  EulerEquation equation;

  Real ComputeStableDt() {
    ampi::tbb_task_scheduler work_scheduler(arena);
    const Real max_s = ampi::reduce_max(work_scheduler, states, [&](const Box& box, int K) {
      Array4<const Real> cons = states.const_array(K);
      Real max_s = 0.0;
      LoopConcurrentOnCpu(box, [&](int i, int j, int k) {
        const Real rho = cons(i, j, k, RHO);
        const Real rhou = cons(i, j, k, RHOU);
        const Real rhov = cons(i, j, k, RHOV);
        const Real rhow = cons(i, j, k, RHOW);
        const Real rhoE = cons(i, j, k, RHOE);
        const Real u = rhou / rho;
        const Real v = rhov / rho;
        const Real w = rhow / rho;
        const Real p = equation.Pressure(rho, rhou, rhov, rhow, rhoE);
        const Real a = std::sqrt(equation.gamma * p / rho);
        const Real abs_a = std::abs(a);
        max_s = std::max(max_s, std::max({std::abs(u), std::abs(v), std::abs(w)}) + std::abs(a));
      });
      return max_s;
    });
    const Geometry& geom = Geom(coarsest_level);
    const Real dx = std::min({geom.CellSize(0), geom.CellSize(1), geom.CellSize(2)});
    return max_s > 0.0 ? dx / max_s : std::numeric_limits<Real>::max();
  }

  // Compute the first order accurate HLLE flux for compressible euler equations
  void ComputeNumericFluxes(
      const Box& box,
      const Array4<Real>& flux,
      const Array4<const Real>& cons,
      Direction dir) const {
    ParallelFor(box, [=, this] AMREX_GPU_DEVICE(int i, int j, int k) {
      // (i,j,k) is face centered and left of cell index (i,j,k)
      Dim3 iR{i, j, k};
      Dim3 iL = Shift(iR, dir, -1);
      // Load left state
      const Real rhoL = cons(iL.x, iL.y, iL.z, RHO);
      const Real rhouL[3] = {
          cons(iL.x, iL.y, iL.z, RHOU), cons(iL.x, iL.y, iL.z, RHOV), cons(iL.x, iL.y, iL.z, RHOW)};
      const Real rhoEL = cons(iL.x, iL.y, iL.z, RHOE);
      const Real uL[3]{rhouL[0] / rhoL, rhouL[1] / rhoL};
      const Real pL = equation.Pressure(rhoL, rhouL[0], rhouL[1], rhouL[2], rhoEL);
      const Real aL = std::sqrt(equation.gamma * pL / rhoL);
      const Real hL = (rhoEL + pL) / rhoL;
      // Load right state
      const Real rhoR = cons(iR.x, iR.y, iR.z, RHO);
      const Real rhouR[3] = {
          cons(iR.x, iR.y, iR.z, RHOU), cons(iR.x, iR.y, iR.z, RHOV), cons(iR.x, iR.y, iR.z, RHOW)};
      const Real rhoER = cons(iR.x, iR.y, iR.z, RHOE);
      const Real uR[3]{rhouR[0] / rhoR, rhouR[1] / rhoR};
      const Real pR = equation.Pressure(rhoR, rhouR[0], rhouR[1], rhouR[2], rhoER);
      const Real aR = std::sqrt(equation.gamma * pR / rhoR);
      const Real hR = (rhoER + pR) / rhoR;
      // Compute Roe averages
      const Real sqRhoL = std::sqrt(rhoL);
      const Real sqRhoR = std::sqrt(rhoR);
      const Real sqRhoSum_inv = 1.0 / (sqRhoL + sqRhoR);
      auto Roe = [=](Real qL, Real qR) {
        return (sqRhoL * qL + sqRhoR * qR) * sqRhoSum_inv;
      };
      const Real roeU[2] = {Roe(uL[0], uR[0]), Roe(uL[1], uR[1])};
      const Real roeH = Roe(hL, hR);
      const Real roeA2 = equation.gm1 * (roeH - 0.5 * (roeU[0] * roeU[0] + roeU[1] * roeU[1]));
      const Real roeA = std::sqrt(roeA2);
      const Real maxA = std::max(aL, aR);
      // Compute Signal velocities
      const int ix = int(dir);
      const int iy = (ix + 1) % 3;
      const int iz = (iy + 1) % 3;
      const Real sL1 = uL[ix] - maxA;
      const Real sL2 = roeU[ix] - roeA;
      const Real sR1 = roeU[ix] + roeA;
      const Real sR2 = uR[ix] + maxA;
      const Real sL = std::min(sL1, sL2);
      const Real sR = std::max(sR1, sR2);
      const Real bL = std::min(sL, 0.0);
      const Real bR = std::max(sR, 0.0);
      const Real db = bR - bL;
      const Real db_positive_inv = db <= 0 ? 1.0 : 1.0 / db;
      // Compute approximative HLLE flux
      auto HLLE = [=](Real fL, Real fR, Real qL, Real qR) {
        return (bR * qR - bL * qL + fL - fR) * db_positive_inv;
      };
      int RHOUs[3] = {RHOU, RHOV, RHOW};
      // clang-format off
      flux(i,j,k,RHO)       = HLLE(rhouL[ix]            , rhouR[ix]            , rhoL     , rhoR);
      flux(i,j,k,RHOUs[ix]) = HLLE(rhouL[ix]*uL[ix] + pL, rhouR[ix]*uR[ix] + pR, rhouL[ix], rhouR[ix]);
      flux(i,j,k,RHOUs[iy]) = HLLE(rhouL[iy]*uL[ix]     , rhouR[iy]*uR[ix]     , rhouL[iy], rhouR[iy]);
      flux(i,j,k,RHOUs[iz]) = HLLE(rhouL[iz]*uL[ix]     , rhouR[iz]*uR[ix]     , rhouL[iz], rhouR[iz]);
      flux(i,j,k,RHOE)      = HLLE(rhoL*hL*uL[ix]       , rhoR*hR*uR[ix]       , rhoEL    , rhoER);
      // clang-format on
    });
  }

  void UpdateConservatively(
      const Box& box,
      const Array4<Real>& cons,
      const Array4<const Real>& flux,
      Real dt_over_dx,
      Direction dir) const {
    ParallelFor(box, int(n_components), [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
      const Dim3 iL{i, j, k};
      const Dim3 iR = Shift(iL, dir, 1);
      cons(i, j, k, n) += dt_over_dx * (flux(iL.x, iL.y, iL.z, n) - flux(iR.x, iR.y, iR.z, n));
    });
  }

  void AsyncAdvanceTime(Real dt) {
    const Real* dx = Geom(coarsest_level).CellSize();
    const Real dt_over_dx[AMREX_SPACEDIM] = {AMREX_D_DECL(dt / dx[0], dt / dx[1], dt / dx[2])};
    auto advance = unifex::bulk_join(unifex::bulk_transform(
        // work_scheduler to schedule work items and comm_scheduler to schedule MPI_WaitAny/All
        // threads
        FillBoundary_async(states, Geom(coarsest_level).periodicity()),
        [this, dt_over_dx](int K, const Box& box) {
          auto advance_dir = [&](Direction dir) {
            const auto dir_v = std::size_t(dir);
            auto csarray = states.const_array(K);
            auto farray = fluxes[dir_v].array(K);
            const Box faces = grow(convert(box, IntVect::TheDimensionVector(dir_v)), dir, -1);
            ComputeNumericFluxes(faces, farray, csarray, dir);
            auto cfarray = fluxes[dir_v].const_array(K);
            auto sarray = states.array(K);
            const Box inner_box = grow(box, -1);
            UpdateConservatively(inner_box, sarray, farray, dt_over_dx[dir_v], dir);
          };
          // first order accurate operator splitting
          advance_dir(Direction::x);
          advance_dir(Direction::y);
          advance_dir(Direction::z);
        },
        unifex::par_unseq));
    // wait here until the above is done for all boxes
    ampi::tbb_task_scheduler work_scheduler(arena);
    auto comm_scheduler = communication_thread_pool.get_scheduler();
    unifex::sync_wait(unifex::with_query_value(
        unifex::with_query_value(std::move(advance), unifex::get_scheduler, work_scheduler),
        ampi::get_comm_scheduler,
        comm_scheduler));
  }

private:
  void ErrorEst(int level, ::amrex::TagBoxArray& tags, Real time_point, int /* ngrow */) override {
    throw std::runtime_error("For simplicity, this example supports only one level.");
  }

  void MakeNewLevelFromScratch(
      int level,
      double,
      const ::amrex::BoxArray& box_array,
      const ::amrex::DistributionMapping& distribution_mapping) override {
    if (level > 0) {
      throw std::runtime_error("For simplicity, this example supports only one level.");
    }
    const IntVect ngrow{AMREX_D_DECL(1, 1, 1)};
    states.define(box_array, distribution_mapping, n_components, ngrow);
    const IntVect ngrow_fs[AMREX_SPACEDIM] = {AMREX_D_DECL(ngrow - ngrow[0] * e_x, ngrow - ngrow[1] * e_y, ngrow - ngrow[2] * e_z)};
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      fluxes[i].define(box_array, distribution_mapping, n_components, ngrow_fs[i]);
    }
  }

  void MakeNewLevelFromCoarse(
      int level,
      double time_point,
      const ::amrex::BoxArray& box_array,
      const ::amrex::DistributionMapping& distribution_mapping) override {
    throw std::runtime_error("For simplicity, this example supports only one level.");
  }

  void RemakeLevel(
      int level,
      double time_point,
      const ::amrex::BoxArray& box_array,
      const ::amrex::DistributionMapping& distribution_mapping) override {
    throw std::runtime_error("For simplicity, this example supports only one level.");
  }

  void ClearLevel(int level) override {
    if (level > 0) {
      throw std::runtime_error("For simplicity, this example supports only one level.");
    }
    states.clear();
    for (MultiFab& fs : fluxes) {
      fs.clear();
    }
  }
};

void my_main() {
}

int main(int argc, char** argv) {
  // Initialize MPI
  const int required = MPI_THREAD_MULTIPLE;
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, required, &provided);
  if (provided < required) {
    std::printf("MPI could not provide enough parallelism.\n");
    return 1;
  }
  unifex::scope_guard mpi_finalize = []() noexcept {
    MPI_Finalize();
  };

  // Initialize AMReX
  amrex::Initialize(
      MPI_COMM_WORLD, std::cout, std::cerr, [](const char* msg) { throw std::runtime_error(msg); });
  unifex::scope_guard amrex_finalize = []() noexcept {
    amrex::Finalize();
  };

  // Call our Application
  my_main();
}
