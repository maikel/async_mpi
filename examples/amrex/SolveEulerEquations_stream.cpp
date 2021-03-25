#include <ampi/amrex/FillBoundary.hpp>

#include <unifex/scope_guard.hpp>
#include <unifex/for_each.hpp>
#include <unifex/sync_wait.hpp>

#include <ranges>

#include <AMReX_NonLocalBC.H>

#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include <unistd.h>

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

  AMREX_GPU_HOST_DEVICE Real
  TotalEnergy(Real rho, Real rhou, Real rhov, Real rhow, Real p) const noexcept {
    const Real rhoE_kin = KineticEnergyDensity(rho, rhou, rhov, rhow);
    const Real rhoE = p * gm1 + rhoE_kin;
    AMREX_ASSERT(rhoE > 0.0);
    return rhoE;
  }
};

Dim3 Shift(Dim3 i, Direction dir, int n) noexcept {
  int array[3]{i.x, i.y, i.z};
  std::size_t dir_s = static_cast<std::size_t>(dir);
  AMREX_ASSERT(dir_s < 3);
  array[dir_s] += n;
  return Dim3{array[0], array[1], array[2]};
}

Box validbox(const BoxArray& ba, const FabArrayBase::TileArray& ta, int index) {
  return ba[ta.indexMap[index]];
}

Box tilebox(
    const BoxArray& ba,
    const FabArrayBase::TileArray& ta,
    int index,
    const IntVect& nodal = IntVect::TheCellVector()) noexcept {
  Box bx(ta.tileArray[index]);
  const IndexType new_typ{nodal};
  if (!new_typ.cellCentered()) {
    bx.setType(new_typ);
    const Box& valid_cc_box = amrex::enclosedCells(validbox(ba, ta, index));
    const IntVect& Big = valid_cc_box.bigEnd();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      if (new_typ.nodeCentered(d)) {  // validbox should also be nodal in d-direction.
        if (bx.bigEnd(d) == Big[d]) {
          bx.growHi(d, 1);
        }
      }
    }
  }
  return bx;
}

Box tilebox(
    const BoxArray& ba,
    const FabArrayBase::TileArray& ta,
    int index,
    const IntVect& nodal,
    const IntVect& ngrow) noexcept {
  Box bx = tilebox(ba, ta, index, nodal);
  const Box& vccbx = validbox(ba, ta, index);
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    if (bx.smallEnd(d) == vccbx.smallEnd(d)) {
      bx.growLo(d, ngrow[d]);
    }
    if (bx.bigEnd(d) >= vccbx.bigEnd(d)) {
      bx.growHi(d, ngrow[d]);
    }
  }
  return bx;
}

Box grownnodaltilebox(
    const BoxArray& ba,
    const FabArrayBase::TileArray& ta,
    int index,
    int dir,
    IntVect const& a_ng) noexcept {
  BL_ASSERT(dir < AMREX_SPACEDIM);
  if (dir < 0)
    return tilebox(ba, ta, index, IntVect::TheNodeVector(), a_ng);
  return tilebox(ba, ta, index, IntVect::TheDimensionVector(dir), a_ng);
}

class EulerAmrCore : public AmrCore {
public:
  enum { coarsest_level };
  enum { RHO, RHOU, RHOV, RHOW, RHOE, n_components };

  MultiFab states{};
  std::array<MultiFab, AMREX_SPACEDIM> fluxes{};

  EulerEquation equation;

  EulerAmrCore(
      const EulerEquation& eq, Geometry const& level_0_geom, AmrInfo const& amr_info = AmrInfo())
    : AmrCore(level_0_geom, amr_info)
    , equation{eq} {
    AmrCore::InitFromScratch(0.0);
    InitData();
  }

  void InitData() {
    const auto problo = Geom(0).ProbLoArray();
    const auto dx = Geom(0).CellSizeArray();
#ifdef AMREX_USE_OMP
#  pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(states, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      Array4<Real> cons = states.array(mfi);
      amrex::ParallelFor(mfi.tilebox(), [&] AMREX_GPU_DEVICE(int i, int j, int k) {
        Real x[] = {AMREX_D_DECL(
            problo[0] + (0.5 + i) * dx[0],
            problo[1] + (0.5 + j) * dx[1],
            problo[2] + (0.5 + k) * dx[2])};
        // const double r2 = AMREX_D_TERM(x[0] * x[0], +x[1] * x[1], +x[2] * x[2]);
        // constexpr double R = 0.1 * 0.1;
        cons(i, j, k, RHO) = 1.0;
        cons(i, j, k, RHOU) = 0.0;
        cons(i, j, k, RHOV) = 0.0;
        cons(i, j, k, RHOW) = 0.0;
        constexpr double p1 = 10.0;
        constexpr double p2 = 1.0;
        cons(i, j, k, RHOE) = x[0] < 0.0 ? equation.TotalEnergy(1.0, 0.0, 0.0, 0.0, p1)
                                         : equation.TotalEnergy(1.0, 0.0, 0.0, 0.0, p2);
      });
    }
  }

  Real ComputeStableDt() {
    Real max_s = ReduceMax(states, IntVect{}, [&](const Box& box, Array4<const Real> cons) {
      Real max_s = 0.0;
      ParallelFor(box, [&max_s, cons, this] AMREX_GPU_DEVICE(int i, int j, int k) {
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
        max_s = std::max(max_s, std::max({std::abs(u), std::abs(v), std::abs(w)}) + std::abs(a));
      });
      return max_s;
    });
    ParallelAllReduce::Max(max_s, ParallelDescriptor::Communicator());
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
      const Real uL[3]{rhouL[0] / rhoL, rhouL[1] / rhoL, rhouL[2] / rhoL};
      const Real pL = equation.Pressure(rhoL, rhouL[0], rhouL[1], rhouL[2], rhoEL);
      const Real aL = std::sqrt(equation.gamma * pL / rhoL);
      const Real hL = (rhoEL + pL) / rhoL;
      // Load right state
      const Real rhoR = cons(iR.x, iR.y, iR.z, RHO);
      const Real rhouR[3] = {
          cons(iR.x, iR.y, iR.z, RHOU), cons(iR.x, iR.y, iR.z, RHOV), cons(iR.x, iR.y, iR.z, RHOW)};
      const Real rhoER = cons(iR.x, iR.y, iR.z, RHOE);
      const Real uR[3]{rhouR[0] / rhoR, rhouR[1] / rhoR, rhouR[2] / rhoR};
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
      const Real roeU[3] = {Roe(uL[0], uR[0]), Roe(uL[1], uR[1]), Roe(uL[2], uR[2])};
      const Real roeH = Roe(hL, hR);
      const Real roeA2 =
          equation.gm1 * (roeH - 0.5 * (roeU[0] * roeU[0] + roeU[1] * roeU[1] + roeU[2] * roeU[2]));
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
      const Real bLbR = bL * bR;
      const Real db = bR - bL;
      const Real db_positive_inv = db <= 0 ? 1.0 : 1.0 / db;
      // Compute approximative HLLE flux
      auto HLLE = [=](Real fL, Real fR, Real qL, Real qR) {
        return (bR * fL - bL * fR + bLbR * (qR - qL)) * db_positive_inv;
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

  void ComputeNumericFluxes(const FabArrayBase::TileArray* tiles, int tile_index, int dir) {
    const Box face_tilebox = tilebox(
        states.boxArray(),
        *tiles,
        tile_index,
        IntVect::TheDimensionVector(dir),
        fluxes[dir].nGrowVect());
    const int K = tiles->indexMap[tile_index];
    AMREX_ASSERT(!states[K].contains_nan());
    auto csarray = states.const_array(K);
    auto farray = fluxes[dir].array(K);
    ComputeNumericFluxes(face_tilebox, farray, csarray, Direction(dir));
    AMREX_ASSERT(!fluxes[dir][K].contains_nan(face_tilebox, 0, n_components));
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

  void UpdateConservatively(const FabArrayBase::TileArray* tiles, int tile_index, double dt_over_dx, int dir) {
    const Box box = enclosedCells(tilebox(
        states.boxArray(),
        *tiles,
        tile_index,
        IntVect::TheDimensionVector(dir),
        fluxes[dir].nGrowVect()));
    const int K = tiles->indexMap[tile_index];
    auto sarray = states.array(K);
    auto cfarray = fluxes[dir].const_array(K);
    UpdateConservatively(box, sarray, cfarray, dt_over_dx, Direction(dir));
  }

  void AdvanceTime(Real dt) {
    // states.FillBoundary(Geom(coarsest_level).periodicity());
    const Real* dx = Geom(coarsest_level).CellSize();
    const Real dt_over_dx[AMREX_SPACEDIM] = {AMREX_D_DECL(dt / dx[0], dt / dx[1], dt / dx[2])};
    const FabArrayBase::TileArray* tiles =
        states.getTileArray(IntVect{FabArrayBase::mfiter_tile_size});
    const int ntiles = tiles->tileArray.size();

    unifex::sync_wait(unifex::for_each(
        FillBoundary_stream(states, Geom(coarsest_level).periodicity()),
        [&](std::vector<int> ready_tiles, const FabArrayBase::TileArray* tiles) {
          const int ntiles = ready_tiles.size();
          for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
#ifdef _OPENMP
#  pragma omp parallel for
#endif
            for (int t = 0; t < ntiles; ++t) {
              const int i = ready_tiles[t];
              ComputeNumericFluxes(tiles, i, dir);
            }
#ifdef _OPENMP
#  pragma omp parallel for
#endif
            for (int t = 0; t < ntiles; ++t) {
              const int i = ready_tiles[t];
              UpdateConservatively(tiles, i, dt_over_dx[dir], dir);
            }
          }
        }));
  }

private:
  void ErrorEst(int, ::amrex::TagBoxArray&, Real, int /* ngrow */) override {
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
    const IntVect ngrow_fs[AMREX_SPACEDIM] = {
        AMREX_D_DECL(ngrow - ngrow[0] * e_x, ngrow - ngrow[1] * e_y, ngrow - ngrow[2] * e_z)};
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      BoxArray faces_dir = box_array;
      faces_dir.convert(IntVect::TheDimensionVector(i));
      fluxes[i].define(faces_dir, distribution_mapping, n_components, ngrow_fs[i]);
    }
  }

  void MakeNewLevelFromCoarse(
      int, double, const ::amrex::BoxArray&, const ::amrex::DistributionMapping&) override {
    throw std::runtime_error("For simplicity, this example supports only one level.");
  }

  void
  RemakeLevel(int, double, const ::amrex::BoxArray&, const ::amrex::DistributionMapping&) override {
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

void WritePlotfiles(const EulerAmrCore& core, double time_point, int step) {
  static const Vector<std::string> varnames{
      "Density", "Momentum_X", "Momentum_Y", "Momentum_Z", "TotalEnergy"};
  // int nlevels = 1;
  std::array<char, 256> x_pbuffer{};
  snprintf(x_pbuffer.data(), x_pbuffer.size(), "AsyncEuler1/plt%09d", step);
  std::string plotfilename{x_pbuffer.data()};
  WriteSingleLevelPlotfile(plotfilename, core.states, varnames, core.Geom(0), time_point, step);
}

void my_main() {
  Box domain(IntVect{}, IntVect{AMREX_D_DECL(63, 63, 63)});
  RealBox real_box{{AMREX_D_DECL(-0.5, -0.5, -0.5)}, {AMREX_D_DECL(0.5, 0.5, 0.5)}};

  Array<int, AMREX_SPACEDIM> all_periodic{AMREX_D_DECL(1, 1, 1)};
  Geometry geom{domain, real_box, CoordSys::cartesian, all_periodic};

  EulerEquation equation{1.4};

  AmrInfo info{};
  info.max_grid_size[0] = IntVect(32);
  EulerAmrCore core(equation, geom, info);
  // WritePlotfiles(core, 0.0, 0);

  double t = 0.0;
  int steps = 0;

  while (t < 1.0 && steps < 50) {
    double dt = 0.3 * core.ComputeStableDt();
    auto start = std::chrono::steady_clock::now();
    core.AdvanceTime(dt);
    auto stop = std::chrono::steady_clock::now();
    std::chrono::nanoseconds diff_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::chrono::microseconds diff_us =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::chrono::milliseconds diff_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    t += dt;
    steps += 1;
    Print() << "[" << diff_ns.count() << "ns : " << diff_us.count() << "us : " << diff_ms.count()
            << "ms] t: " << t << ", step: " << steps << '\n';
    // WritePlotfiles(core, t, steps);
  }
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

  // {
  //   volatile int i = 0;
  //   char hostname[256];
  //   gethostname(hostname, sizeof(hostname));
  //   printf("PID %d on %s ready for attach\n", getpid(), hostname);
  //   fflush(stdout);
  //   while (0 == i)
  //       sleep(5);
  // }

  // Call our Application
  my_main();
}
