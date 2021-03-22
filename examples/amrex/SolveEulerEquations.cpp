#include <ampi/amrex/FillBoundary.hpp>

#include <ampi/bulk_sequence.hpp>
#include <ampi/for_each.hpp>
#include <ampi/bulk_on.hpp>
#include <ampi/tbb_task_scheduler.hpp>

#include <unifex/single_thread_context.hpp>
#include <unifex/bulk_join.hpp>
#include <unifex/bulk_transform.hpp>
#include <unifex/sync_wait.hpp>

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
  enum { RHO, RHOU, RHOV, RHOE, n_components };

  tbb::task_arena arena;
  MultiFab states;
  std::vector<MultiFab> fluxes;

  EulerEquation equation;

  Real ComputeStableDt() const {
    const Real max_s = ReduceMax(tbb_scheduler, states, [&](const Box& box, int K) {
      Array4<const Real> cons = states.const_array(K);
      Real max_s = 0.0;
      LoopConcurrentOnCpu(box, [&](int i, int j, int k) {
        const Real rho = cons(i, j, k, RHO);
        const Real rhou = cons(i, j, k, RHOU);
        const Real rhov = cons(i, j, k, RHOV);
        const Real rhow = cons(i, j, k, RHOV);
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

  void ComputeNumericFluxes(
      const Box& box,
      const Array4<Real>& flux,
      const Array4<const Real>& cons,
      Direction dir) const {
    ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      // (i,j,k) is face centered and left of cell index (i,j,k)
      Dim3 iR{i, j, k};
      Dim3 iL = Shift(iR, dir, -1);
      // Load left state
      const Real rhoL = cons(iL.x, iL.y, iL.z, RHO);
      const Real rhouL[2] = {cons(iL.x, iL.y, iL.z, RHOU), cons(iL.x, iL.y, iL.z, RHOV)};
      const Real rhoEL = cons(iL.x, iL.y, iL.z, RHOE);
      const Real uL[2]{rhouL[0] / rhoL, rhouL[1] / rhoL};
      const Real pL = equation.Pressure(rhoL, rhouL[0], rhouL[1], rhoEL);
      const Real aL = std::sqrt(equation.gamma * pL / rhoL);
      const Real hL = (rhoEL + pL) / rhoL;
      // Load right state
      const Real rhoR = cons(iR.x, iR.y, iR.z, RHO);
      const Real rhouR[2] = {cons(iR.x, iR.y, iR.z, RHOU), cons(iR.x, iR.y, iR.z, RHOV)};
      const Real rhoER = cons(iR.x, iR.y, iR.z, RHOE);
      const Real uR[2]{rhouR[0] / rhoR, rhouR[1] / rhoR};
      const Real pR = equation.Pressure(rhoR, rhouR[0], rhouR[1], rhoER);
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
      const int iy = 1 - ix;
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
      int RHOUs[2] = {RHOU, RHOV};
      // clang-format off
      flux(i,j,k,RHO)       = HLLE(rhouL[ix]            , rhouR[ix]            , rhoL     , rhoR);
      flux(i,j,k,RHOUs[ix]) = HLLE(rhouL[ix]*uL[ix] + pL, rhouR[ix]*uR[ix] + pR, rhouL[ix], rhouR[ix]);
      flux(i,j,k,RHOUs[iy]) = HLLE(rhouL[iy]*uL[ix]     , rhouR[iy]*uR[ix]     , rhouL[iy], rhouR[iy]);
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

  void AdvanceTime(double dt) {
    auto advance = unifex::bulk_join(unifex::bulk_transform(
      // tbb_scheduler for work and comm_scheduler for MPI_WaitAny
      FillBoundary(tbb_scheduler, comm_scheduler, states), 
      [this, dt_over_dx](const Box& box, int K) {
        auto advance_dir = [&](Direction dir) {
          auto csarray = states.const_array(K);
          auto farray = fluxes[dir_v].array(K);
          Box faces = shrink(convert(box, unit(dir)), 1, unit(dir));
          ComputeNumericFluxes(faces, farray, csarray, dir);
          auto cfarray = fluxes[dir_v].const_array(K);
          auto sarray = states.array(K);
          UpdateConservatively(inner_box, sarray, farray, dt_over_dx[dir_v], dir);
        };
        advance_dir(Direction::x);
        advance_dir(Direction::y);
        advance_dir(Direction::z);
      }, unifex::par_unseq));
    // wait here until the above is done for all boxes
    unifex::sync_wait(std::move(advance));
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
    const IntVect ngrow{AMREX_D_DECL(1, 1, 0)};
    states.define(box_array, distribution_mapping, n_components, ngrow);
    const IntVect ngrow_fs[2] = {ngrow[0] * e_y, ngrow[1] * e_x};
    AMREX_ASSERT(rank < 2);
    for (int i = 0; i < rank; ++i) {
      fluxes.emplace_back(box_array, distribution_mapping, n_components, ngrow_fs[i]);
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
