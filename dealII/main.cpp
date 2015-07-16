#ifndef _MSC_VER
#define TYPENAME typename
#else
#define TYPENAME 
#endif
#define DIM 3
#define DEPTH 6
#define MAGNET_SIZE 2
#define AIR_LAYER_THICKNESS 1
#define INIT_REF_NUM 10

const bool NO_MOVEMENT_INDUCED_FORCE = true;
const bool NO_EXT_CURR_DENSITY_FORCE = false;

const bool PRINT_ALGEBRA = false, PRINT_INIT_SLN = true;

#include <deal.II/base/point.h>
#include <vector>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/full_matrix.h>    
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>


#include <fstream>
#include <iostream>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>
#include "tbb/tbb.h"

#pragma region setup

const dealii::Point<DIM> p1Mag(0., 0., 0.);
const dealii::Point<DIM> p2Mag(1., 1., 1.);
const std::vector<unsigned int> refinementsMag({ INIT_REF_NUM, INIT_REF_NUM, DEPTH });

const dealii::Point<DIM> singleLayerThicknessMag((p2Mag(0) - p1Mag(0)) / ((double)refinementsMag[0]), (p2Mag(1) - p1Mag(1)) / ((double)refinementsMag[1]), (p2Mag(2) - p1Mag(2)) / ((double)refinementsMag[2]));
const double flowChannelOffsetXY[2] = { ((p2Mag[0] - p1Mag[0]) / (INIT_REF_NUM)) * (AIR_LAYER_THICKNESS + MAGNET_SIZE) + p1Mag[0], ((p2Mag[1] - p1Mag[1]) / (INIT_REF_NUM)) * (AIR_LAYER_THICKNESS + MAGNET_SIZE) + p1Mag[1] };

const dealii::Point<DIM> p1Flow(flowChannelOffsetXY[0], flowChannelOffsetXY[1], 0.);
const dealii::Point<DIM> p2Flow(1. - flowChannelOffsetXY[0], 1. - flowChannelOffsetXY[1], 1.);
const std::vector<unsigned int> refinementsFlow({ INIT_REF_NUM - AIR_LAYER_THICKNESS - MAGNET_SIZE, INIT_REF_NUM - AIR_LAYER_THICKNESS - MAGNET_SIZE, DEPTH });

// boundary id
const unsigned int BOUNDARY_FRONT = 1;
const unsigned int BOUNDARY_RIGHT = 2;
const unsigned int BOUNDARY_BACK = 3;
const unsigned int BOUNDARY_LEFT = 4;
const unsigned int BOUNDARY_BOTTOM = 5;
const unsigned int BOUNDARY_TOP = 6;
const unsigned int BOUNDARY_ELECTRODES = 8;

std::vector<unsigned int> velocityDirichletMarkers({ BOUNDARY_LEFT, BOUNDARY_RIGHT, BOUNDARY_BOTTOM, BOUNDARY_TOP });
std::vector<unsigned int> magnetismDirichletMarkers({ BOUNDARY_BOTTOM, BOUNDARY_TOP, BOUNDARY_LEFT, BOUNDARY_RIGHT });
std::vector<unsigned int> currentDirichletMarkers({ BOUNDARY_ELECTRODES, BOUNDARY_ELECTRODES });

const bool INLET_VELOCITY_FIXED = false;
const unsigned int INLET_VELOCITY_FIXED_BOUNDARY = BOUNDARY_BACK;
const double INLET_VELOCITY_AMPLITUDE = 10.0;

const unsigned int POLYNOMIAL_DEGREE_MAG = 1;
const unsigned int POLYNOMIAL_DEGREE_E = 2;

const double MU = 1.2566371e-6;
const double MU_R = 1.;

// material id 
const unsigned int MARKER_AIR = 0;
const unsigned int MARKER_MAGNET = 1;
const unsigned int MARKER_FLUID = 2;
const unsigned int MARKER_ELECTRODE = 3;

// These are according to components
const double B_R[3] = { 0., -1., 0. };

const double SIGMA = 3.e6;
const double DENSITY = 8.e4;

const double J_EXT_VAL = 1.e5;
double J_EXT(int marker, int component, dealii::Point<DIM> p)
{
  if (marker == MARKER_FLUID && component == 0)
    return J_EXT_VAL;
  return 0.;
}

const double REYNOLDS = 5.;

const double NEWTON_DAMPING = .8;
const int NEWTON_ITERATIONS = 100;
const double NEWTON_RESIDUAL_THRESHOLD = 1e-8;

const double TIME_STEP = 1.e-3;
const double T_END = 1.;

dealii::Tensor<1, 3> curl(dealii::Tensor<1, 3>& gradient_0, dealii::Tensor<1, 3>& gradient_1, dealii::Tensor<1, 3>& gradient_2)
{
  dealii::Tensor<1, 3> result;
  result[0] = gradient_2[1] - gradient_1[2];
  result[1] = gradient_0[2] - gradient_2[0];
  result[2] = gradient_1[0] - gradient_0[1];
  return result;
}

dealii::Tensor<1, 3> custom_cross_product(dealii::Tensor<1, 3>& left, dealii::Tensor<1, 3>& right)
{
  dealii::Tensor<1, 3> result;
  dealii::cross_product(result, left, right);
  return result;
}

#pragma endregion

namespace MHD
{
  using namespace dealii;

#pragma region assemblyData
  class AssemblyScratchData
  {
  public:
    AssemblyScratchData(const dealii::hp::FECollection<DIM> &feCollectionFlow,
      const dealii::hp::MappingCollection<DIM> &mappingCollectionFlow,
      const dealii::hp::QCollection<DIM> &quadratureFormulasFlow, const dealii::hp::FECollection<DIM> &feCollectionMag,
      const dealii::hp::MappingCollection<DIM> &mappingCollectionMag,
      const dealii::hp::QCollection<DIM> &quadratureFormulasMag);
    AssemblyScratchData(const AssemblyScratchData &scratch_data);

    dealii::hp::FEValues<DIM> hp_fe_values_Flow;
    dealii::hp::FEValues<DIM> hp_fe_values_Mag;
  };

  class AssemblyCopyData
  {
  public:
    AssemblyCopyData();

    bool isAssembled;

    dealii::FullMatrix<double> cell_matrix;
    dealii::Vector<double> cell_rhs;

    std::vector<dealii::types::global_dof_index> local_dof_indices;
  };

  AssemblyScratchData::AssemblyScratchData(const dealii::hp::FECollection<DIM> &feCollectionFlow,
    const dealii::hp::MappingCollection<DIM> &mappingCollectionFlow,
    const dealii::hp::QCollection<DIM> &quadratureFormulasFlow, const dealii::hp::FECollection<DIM> &feCollectionMag,
    const dealii::hp::MappingCollection<DIM> &mappingCollectionMag,
    const dealii::hp::QCollection<DIM> &quadratureFormulasMag)
    :
    hp_fe_values_Flow(mappingCollectionFlow, feCollectionFlow, quadratureFormulasFlow, dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values),
    hp_fe_values_Mag(mappingCollectionMag, feCollectionMag, quadratureFormulasMag, dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values)
  {}

  AssemblyScratchData::AssemblyScratchData(const AssemblyScratchData &scratch_data)
    :
    hp_fe_values_Flow(scratch_data.hp_fe_values_Flow.get_mapping_collection(),
    scratch_data.hp_fe_values_Flow.get_fe_collection(),
    scratch_data.hp_fe_values_Flow.get_quadrature_collection(),
    dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values),
    hp_fe_values_Mag(scratch_data.hp_fe_values_Mag.get_mapping_collection(),
    scratch_data.hp_fe_values_Mag.get_fe_collection(),
    scratch_data.hp_fe_values_Mag.get_quadrature_collection(),
    dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values)
  {}

  AssemblyCopyData::AssemblyCopyData()
    : isAssembled(false), cell_matrix(0), cell_rhs(0)
  {}

#pragma endregion 

#pragma region solver

  // Forward declaration.
  namespace Mag
  {
    class Solver;
  }

  namespace Flow
  {
    class Solver
    {
    public:
      Solver();
      ~Solver();

      void init_discretization(Mag::Solver* magSolver);
      void solveOneStep(Vector<double> previousMagSln, int iteration);
      void output_results(int time_iteration, int inner_iteration);

      // Mesh
      void applyDirichletToInitialSln();
      void add_markers(Triangulation<DIM>::cell_iterator cell);

      // High level one step - set up
      void setup_system(const bool initial_step);
      // High level one step - assemble
      void assemble_system();
      // High level one step - solve
      void solveAlgebraicSystem(int inner_iteration);

      // Assembling
      void localAssembleSystem(const dealii::hp::DoFHandler<DIM>::active_cell_iterator &iter,
        AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data);

      // Assembling
      void copyLocalToGlobal(const AssemblyCopyData &copy_data);

      // Assembling
      void finishAssembling();

      // Data
      Triangulation<DIM>   triangulation;
      hp::DoFHandler<DIM>      dof_handler;
      Mag::Solver*      magSolver;

      dealii::hp::FECollection<DIM> feCollection;
      dealii::hp::MappingCollection<DIM> mappingCollection;
      dealii::hp::QCollection<DIM> qCollection;
      dealii::hp::QCollection<DIM - 1> qCollectionFace;

      ConstraintMatrix     hanging_node_constraints;

      SparsityPattern      sparsity_pattern;
      SparseMatrix<double> system_matrix;

      Vector<double>       present_solution;
      Vector<double>       previous_sln_Flow;
      Vector<double>       previous_sln_Mag;
      Vector<double>       newton_update;
      Vector<double>       system_rhs;

      dealii::SparseDirectUMFPACK direct_CustomSolver;
    };
  }

  namespace Mag
  {
    class Solver
    {
    public:
      Solver();
      ~Solver();

      void init_discretization(Flow::Solver* flowSolver);
      void solveOneStep(Vector<double> previousFlowSln, int iteration);
      void output_results(int inner_iteration);

      // Mesh
      void applyDirichletToInitialSln();
      void add_markers(Triangulation<DIM>::cell_iterator cell);

      // High level one step - set up
      void setup_system(const bool initial_step);
      // High level one step - assemble
      void assemble_system();
      // High level one step - solve
      void solveAlgebraicSystem();

      // Assembling
      void localAssembleSystem(const dealii::hp::DoFHandler<DIM>::active_cell_iterator &iter,
        AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data);

      // Assembling
      void copyLocalToGlobal(const AssemblyCopyData &copy_data);

      // Assembling
      void finishAssembling();

      // Data
      Triangulation<DIM>   triangulation;
      hp::DoFHandler<DIM>      dof_handler;
      Flow::Solver*      flowSolver;

      dealii::hp::FECollection<DIM> feCollection;
      dealii::hp::MappingCollection<DIM> mappingCollection;
      dealii::hp::QCollection<DIM> qCollection;
      dealii::hp::QCollection<DIM - 1> qCollectionFace;

      ConstraintMatrix     hanging_node_constraints;

      SparsityPattern      sparsity_pattern;
      SparseMatrix<double> system_matrix;

      Vector<double>       present_solution;
      Vector<double> previous_sln_Flow;
      Vector<double>       system_rhs;

      dealii::SparseDirectUMFPACK direct_CustomSolver;
    };
  }

  class Solver
  {
  public:
    Solver();
    ~Solver();
    void run();

    Flow::Solver* flowSolver;
    Mag::Solver* magSolver;
  };
#pragma endregion

#pragma region initialization
  namespace Flow
  {
    Solver::Solver() : dof_handler(triangulation)
    {
      std::vector<const dealii::FiniteElement<DIM> *> fes;
      std::vector<unsigned int> multiplicities;

      // Velocity
      fes.push_back(new dealii::FE_Q<DIM>(2));
      multiplicities.push_back(DIM);

      // Pressure
      fes.push_back(new dealii::FE_Q<DIM>(1));
      multiplicities.push_back(1);

      feCollection.push_back(dealii::FESystem<DIM, DIM>(fes, multiplicities));

      mappingCollection.push_back(dealii::MappingQ<DIM>(1, true));

      qCollection.push_back(dealii::QGauss<DIM>(3 * std::max<unsigned int>(POLYNOMIAL_DEGREE_MAG, 2)));
      qCollectionFace.push_back(dealii::QGauss<DIM - 1>(3 * std::max<unsigned int>(POLYNOMIAL_DEGREE_MAG, 2)));
    }

    Solver::~Solver()
    {
      dof_handler.clear();
    }

    void Solver::init_discretization(Mag::Solver* magSolver)
    {
      // Coupling
      this->magSolver = magSolver;

      // Mesh
      GridGenerator::subdivided_hyper_rectangle(triangulation, refinementsFlow, p1Flow, p2Flow);

      Triangulation<DIM>::cell_iterator
        cell = triangulation.begin(),
        endc = triangulation.end();
      for (; cell != endc; ++cell)
      {
        this->add_markers(cell);
      }

      this->setup_system(true);
      this->applyDirichletToInitialSln();
    }

    void Solver::add_markers(Triangulation<DIM>::cell_iterator cell)
    {
      // Volumetric.
      cell->set_material_id(MARKER_FLUID);

      // Surface.
      for (unsigned int face_number = 0; face_number < GeometryInfo<DIM>::faces_per_cell; ++face_number)
      {
        if (std::fabs(cell->face(face_number)->center()(2) - p1Flow(2)) < 1e-12)
          cell->set_material_id(BOUNDARY_BACK);

        if (std::fabs(cell->face(face_number)->center()(0) - p1Flow(0)) < 1e-12)
          cell->set_material_id(BOUNDARY_LEFT);

        if (std::fabs(cell->face(face_number)->center()(2) - p2Flow(2)) < 1e-12)
          cell->set_material_id(BOUNDARY_FRONT);

        if (std::fabs(cell->face(face_number)->center()(0) - p2Flow(0)) < 1e-12)
          cell->set_material_id(BOUNDARY_RIGHT);

        if (std::fabs(cell->face(face_number)->center()(1) - p1Flow(1)) < 1e-12)
          cell->set_material_id(BOUNDARY_BOTTOM);

        if (std::fabs(cell->face(face_number)->center()(1) - p2Flow(1)) < 1e-12)
          cell->set_material_id(BOUNDARY_TOP);
      }
    }

    void Solver::setup_system(const bool initial_step)
    {
      if (initial_step)
      {
        dof_handler.distribute_dofs(feCollection);

        dealii::DoFRenumbering::component_wise(dof_handler);

        present_solution.reinit(dof_handler.n_dofs());

        hanging_node_constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler,
          hanging_node_constraints);
        hanging_node_constraints.close();
      }

      newton_update.reinit(dof_handler.n_dofs());
      system_rhs.reinit(dof_handler.n_dofs());

      CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, c_sparsity);
      hanging_node_constraints.condense(c_sparsity);
      sparsity_pattern.copy_from(c_sparsity);
      system_matrix.reinit(sparsity_pattern);

      // Previous time-level solution.
      this->previous_sln_Flow = this->present_solution;
    }
  }

  namespace Mag
  {
    Solver::Solver() : dof_handler(triangulation)
    {
      std::vector<const dealii::FiniteElement<DIM> *> fes;
      std::vector<unsigned int> multiplicities;

      // A
      fes.push_back(new dealii::FE_Q<DIM>(POLYNOMIAL_DEGREE_MAG));
      multiplicities.push_back(DIM);
      feCollection.push_back(dealii::FESystem<DIM, DIM>(fes, multiplicities));

      mappingCollection.push_back(dealii::MappingQ<DIM>(1, true));

      qCollection.push_back(dealii::QGauss<DIM>(3 * std::max<unsigned int>(POLYNOMIAL_DEGREE_MAG, 2)));
      qCollectionFace.push_back(dealii::QGauss<DIM - 1>(3 * std::max<unsigned int>(POLYNOMIAL_DEGREE_MAG, 2)));
    }

    Solver::~Solver()
    {
      dof_handler.clear();
    }

    void Solver::init_discretization(Flow::Solver* flowSolver)
    {
      this->flowSolver = flowSolver;

      // Mesh
      GridGenerator::subdivided_hyper_rectangle(triangulation, refinementsMag, p1Mag, p2Mag);

      Triangulation<DIM>::cell_iterator
        cell = triangulation.begin(),
        endc = triangulation.end();
      for (; cell != endc; ++cell)
      {
        this->add_markers(cell);
      }

      this->setup_system(true);
      this->applyDirichletToInitialSln();
    }

    void Solver::add_markers(Triangulation<DIM>::cell_iterator cell)
    {
      // Volumetric.
      // 0 Left, 1 Right, 2 Bottom, 3 Top, 4 Front, 5 Back
      int layerFromEdge[6] = { 0, 0, 0, 0, 0, 0 };
      int comparedCoordinate[6] = { 0, 0, 1, 1, 2, 2 };
      double comparedValue[6] = { p1Mag(0), p2Mag(0), p1Mag(1), p2Mag(1), p1Mag(2), p2Mag(2) };

      for (unsigned int face_number = 0; face_number < GeometryInfo<DIM>::faces_per_cell; ++face_number)
      {
        layerFromEdge[face_number] = std::round(std::fabs(cell->face(face_number)->center()(comparedCoordinate[face_number]) - comparedValue[face_number]) / singleLayerThicknessMag(comparedCoordinate[face_number]));
      }

      cell->set_material_id(MARKER_AIR);
      bool fluid = true;
      for (unsigned int i = 0; i < GeometryInfo<DIM>::faces_per_cell; ++i)
      {
        if (i == 4 || i == 5)
          continue;
        if (layerFromEdge[i] < AIR_LAYER_THICKNESS + MAGNET_SIZE)
          fluid = false;
      }
      if (fluid)
        cell->set_material_id(MARKER_FLUID);

      bool magnet = true;
      for (unsigned int i = 0; i < GeometryInfo<DIM>::faces_per_cell; ++i)
      {
        if (i == 4 || i == 5)
          continue;

        if (i == 0 || i == 1) {
          if (layerFromEdge[i] < AIR_LAYER_THICKNESS + MAGNET_SIZE)
            magnet = false;
        }

        if (i == 2 || i == 3) {
          if (layerFromEdge[i] < AIR_LAYER_THICKNESS)
            magnet = false;
        }

        if (cell->material_id() == MARKER_FLUID)
          magnet = false;
      }
      if (magnet)
        cell->set_material_id(MARKER_MAGNET);

      // Surface.
      for (unsigned int face_number = 0; face_number < GeometryInfo<DIM>::faces_per_cell; ++face_number)
      {
        if (std::fabs(cell->face(face_number)->center()(2) - p1Mag(2)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator(BOUNDARY_BACK);

        if (std::fabs(cell->face(face_number)->center()(0) - p1Mag(0)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator(BOUNDARY_LEFT);

        if (std::fabs(cell->face(face_number)->center()(2) - p2Mag(2)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator(BOUNDARY_FRONT);

        if (std::fabs(cell->face(face_number)->center()(0) - p2Mag(0)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator(BOUNDARY_RIGHT);

        if (std::fabs(cell->face(face_number)->center()(1) - p1Mag(1)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator(BOUNDARY_BOTTOM);

        if (std::fabs(cell->face(face_number)->center()(1) - p2Mag(1)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator(BOUNDARY_TOP);
      }
    }

    void Solver::setup_system(const bool initial_step)
    {
      if (initial_step)
      {
        dof_handler.distribute_dofs(feCollection);

        dealii::DoFRenumbering::component_wise(dof_handler);

        present_solution.reinit(dof_handler.n_dofs());

        hanging_node_constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler,
          hanging_node_constraints);
        hanging_node_constraints.close();
      }

      system_rhs.reinit(dof_handler.n_dofs());

      CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, c_sparsity);
      hanging_node_constraints.condense(c_sparsity);
      sparsity_pattern.copy_from(c_sparsity);
      system_matrix.reinit(sparsity_pattern);
    }
  }

#pragma region BC_values

  class BoundaryValuesWall : public Function < DIM >
  {
  public:
    BoundaryValuesWall(int componentCount) : Function<DIM>(componentCount) {}

    virtual double value(const Point<DIM>   &p, const unsigned int  component = 0) const;
  };

  double BoundaryValuesWall::value(const Point<DIM> &p, const unsigned int component) const
  {
    return 0.;
  }

  class BoundaryValuesInlet : public Function < DIM >
  {
  public:
    BoundaryValuesInlet(int componentCount) : Function<DIM>(componentCount) {}

    virtual double value(const Point<DIM>   &p, const unsigned int  component = 0) const;
  };

  double BoundaryValuesInlet::value(const Point<DIM> &p, const unsigned int component) const
  {
    if (component == 2)
      return INLET_VELOCITY_AMPLITUDE * ((p(0) - p1Flow(0)) * (p2Flow(0) - p(0))) * ((p(1) - p1Flow(1)) * (p2Flow(1) - p(1)));
    else
      return 0;
  }

  namespace Flow
  {
    void Solver::applyDirichletToInitialSln()
    {
      std::map<types::global_dof_index, double> boundary_values;

      ComponentMask velocity_mask(DIM + 1, false);
      for (int i = 0; i < DIM; i++)
        velocity_mask.set(i, true);

      for (std::vector<unsigned int>::iterator it = velocityDirichletMarkers.begin(); it != velocityDirichletMarkers.end(); ++it)
      {
        VectorTools::interpolate_boundary_values(dof_handler, *it, BoundaryValuesWall(DIM + 1), boundary_values, velocity_mask);
        for (std::map<types::global_dof_index, double>::const_iterator p = boundary_values.begin(); p != boundary_values.end(); ++p)
          present_solution(p->first) = p->second;
      }
      if (INLET_VELOCITY_FIXED)
      {
        VectorTools::interpolate_boundary_values(dof_handler, BOUNDARY_BOTTOM, BoundaryValuesInlet(DIM + 1), boundary_values, velocity_mask);
        for (std::map<types::global_dof_index, double>::const_iterator p = boundary_values.begin(); p != boundary_values.end(); ++p)
          present_solution(p->first) = p->second;
      }
    }
  }

  namespace Mag
  {
    void Solver::applyDirichletToInitialSln()
    {
      std::map<types::global_dof_index, double> boundary_values;

      for (std::vector<unsigned int>::iterator it = magnetismDirichletMarkers.begin(); it != magnetismDirichletMarkers.end(); ++it)
      {
        VectorTools::interpolate_boundary_values(dof_handler, *it, BoundaryValuesWall(DIM), boundary_values);
        for (std::map<types::global_dof_index, double>::const_iterator p = boundary_values.begin(); p != boundary_values.end(); ++p)
          present_solution(p->first) = p->second;
      }
    }
  }

#pragma endregion

#pragma endregion

#pragma region assembling - magnetism
  namespace Mag
  {
    void Solver::localAssembleSystem(const dealii::hp::DoFHandler<DIM>::active_cell_iterator &cell, AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data)
    {
      scratch_data.hp_fe_values_Mag.reinit(cell);
      const dealii::FEValues<DIM> &fe_values_Mag = scratch_data.hp_fe_values_Mag.get_present_fe_values();

      const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      const unsigned int n_q_points = fe_values_Mag.n_quadrature_points;

      copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      copy_data.cell_matrix = 0;
      copy_data.cell_rhs.reinit(dofs_per_cell);
      copy_data.cell_rhs = 0;

      std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      std::vector<dealii::Vector<double> > prev_values_Flow(n_q_points, dealii::Vector<double>(DIM + 1));
      std::vector<std::vector<dealii::Tensor<1, DIM> > > prev_gradients_Flow(n_q_points, std::vector<dealii::Tensor<1, DIM> >(DIM + 1));

      std::vector<dealii::Vector<double> > prev_values_Mag(n_q_points, dealii::Vector<double>(DIM));
      std::vector<std::vector<dealii::Tensor<1, DIM> > > prev_gradients_Mag(n_q_points, std::vector<dealii::Tensor<1, DIM> >(DIM));

      std::vector<int> components(dofs_per_cell);

      fe_values_Mag.get_function_values(present_solution, prev_values_Mag);
      fe_values_Mag.get_function_gradients(present_solution, prev_gradients_Mag);

      /*
      // Previous flow values
      // - find the element on the second mesh.
      dealii::hp::DoFHandler<DIM>::active_cell_iterator &cellFlow = GridTools::find_active_cell_around_point(this->flowSolver->dof_handler, cell->center());
      scratch_data.hp_fe_values_Flow.reinit(cellFlow);
      const dealii::FEValues<DIM> &fe_values_Flow = scratch_data.hp_fe_values_Flow.get_present_fe_values();
      // - get the values.
      fe_values_Flow.get_function_values(this->previous_sln_Flow, prev_values_Flow);
      fe_values_Flow.get_function_gradients(this->previous_sln_Flow, prev_gradients_Flow);
      */

      // curl A from the previous iteration.
      std::vector<dealii::Tensor<1, DIM> > C(n_q_points);
      for (int i = 0; i < n_q_points; i++)
        C[i] = curl(prev_gradients_Mag[i][0], prev_gradients_Mag[i][1], prev_gradients_Mag[i][2]);

      std::vector<std::vector<double> > shape_value(dofs_per_cell, std::vector<double>(n_q_points));
      std::vector<double> JxW(n_q_points);
      std::vector<std::vector<dealii::Tensor<1, DIM> > > shape_grad(dofs_per_cell, std::vector<dealii::Tensor<1, DIM> >(n_q_points));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        components[i] = cell->get_fe().system_to_component_index(i).first;
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          shape_value[i][q_point] = fe_values_Mag.shape_value(i, q_point);
          shape_grad[i][q_point] = fe_values_Mag.shape_grad(i, q_point);
        }
      }
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        JxW[q_point] = fe_values_Mag.JxW(q_point);

      // Volumetric marker
      unsigned int marker = cell->material_id();

      // Geometrical points
      std::vector<dealii::Point<DIM> > points;
      points.reserve(dealii::DoFTools::max_dofs_per_face(dof_handler));
      points = fe_values_Mag.get_quadrature_points();

      // distribute local to global matrix
      copy_data.local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(copy_data.local_dof_indices);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        dealii::Point<DIM> q_p = points[q_point];
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            // Laplace
            if (components[i] == components[j])
            {
              copy_data.cell_matrix(i, j) += shape_grad[i][q_point]
                * shape_grad[j][q_point]
                * JxW[q_point]
                / (MU * MU_R);
            }

            // Remanent induction.
            if (marker == MARKER_MAGNET)
            {
              /* TODO - KDYZ SE TAM DA TENTO (ASI SPRAVNY) VYRAZ, A PRIDA SE VYRAZ PRO NEUMANNA DALE, TAK TO NEJDE
              if (components[i] == 0) {
                copy_data.cell_rhs(i) -= (B_R[2] * shape_grad[i][q_point][1] - B_R[1] * shape_grad[i][q_point][2])
                  * JxW[q_point]
                  / (MU * MU_R);
              }
              */
              if (components[i] == 1) {
                copy_data.cell_rhs(i) -= (B_R[0] * shape_grad[i][q_point][2] - B_R[2] * shape_grad[i][q_point][0])
                  * JxW[q_point]
                  / (MU * MU_R);
              }

              if (components[i] == 2) {
                copy_data.cell_rhs(i) -= (B_R[1] * shape_grad[i][q_point][0] - B_R[0] * shape_grad[i][q_point][1])
                  * JxW[q_point]
                  / (MU * MU_R);
              }
            }
          }
        }
      }

      // Remanent induction - SURFACE.
      /* TODO - KDYZ SE TAM DA TENTO (ASI SPRAVNY) VYRAZ, A PRIDA SE VYRAZ PRO REMANENTNI INDUKCI V (component == 0), TAK TO NEJDE - proto je ted na pravou stranu (component == 0) dana nula
      if (marker == MARKER_MAGNET)
      {
        dealii::hp::FEFaceValues<DIM> hp_fe_face_values(mappingCollection, feCollection, qCollectionFace, dealii::update_quadrature_points | dealii::update_values | dealii::update_JxW_values);

        std::cout << "Cell: " << cell->center()(0) << ", " << cell->center()(1) << ", " << cell->center()(2) << std::endl;
        for (unsigned int face = 0; face < dealii::GeometryInfo<DIM>::faces_per_cell; ++face)
        {
          if (cell->face(face)->boundary_indicator() == BOUNDARY_FRONT || cell->face(face)->boundary_indicator() == BOUNDARY_BACK)
          {
            std::cout << "Face: " << cell->face(face)->center()(0) << ", " << cell->face(face)->center()(1) << ", " << cell->face(face)->center()(2) << std::endl;

            hp_fe_face_values.reinit(cell, face);

            const dealii::FEFaceValues<DIM> &fe_face_values = hp_fe_face_values.get_present_fe_values();
            const unsigned int n_face_q_points = fe_face_values.n_quadrature_points;

            std::vector<std::vector<double> > shape_face_value = std::vector<std::vector<double> >(dofs_per_cell);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              shape_face_value[i].resize(n_face_q_points);
              for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                shape_face_value[i][q_point] = fe_face_values.shape_value(i, q_point);
            }

            std::vector<double> shape_face_JxW(n_face_q_points);
            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
              shape_face_JxW[q_point] = fe_face_values.JxW(q_point);

            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                double normal_z = (cell->face(face)->boundary_indicator() == BOUNDARY_FRONT) ? 1.0 : -1.0;

                if (components[i] == 0) {
                  copy_data.cell_rhs(i) += B_R[1] * normal_z * shape_face_value[i][q_point]
                    * shape_face_JxW[q_point]
                    / (MU * MU_R);
                }
              }
            }
          }
        }
      }
      */

      // distribute local to global matrix
      copy_data.local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(copy_data.local_dof_indices);

      copy_data.isAssembled = true;
    }

    void Solver::assemble_system()
    {
      system_matrix = 0;
      system_rhs = 0;

      dealii::hp::FEValues<DIM> hp_fe_values(feCollection, qCollection, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

      dealii::hp::DoFHandler<DIM>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

      dealii::WorkStream::run(cell, endc,
        *this,
        &Solver::localAssembleSystem,
        &Solver::copyLocalToGlobal,
        AssemblyScratchData(this->flowSolver->feCollection, this->flowSolver->mappingCollection, this->flowSolver->qCollection, this->feCollection, this->mappingCollection, this->qCollection),
        AssemblyCopyData());

      this->finishAssembling();
    }

    void Solver::copyLocalToGlobal(const AssemblyCopyData &copy_data)
    {
      if (copy_data.isAssembled)
      {
        hanging_node_constraints.distribute_local_to_global(copy_data.cell_matrix,
          copy_data.cell_rhs,
          copy_data.local_dof_indices,
          system_matrix,
          system_rhs);
      }
    }

    void Solver::finishAssembling()
    {
      // Finally, we remove hanging nodes from the system and apply zero
      // $\delta u^n$:
      hanging_node_constraints.condense(system_matrix);
      hanging_node_constraints.condense(system_rhs);

      std::map<types::global_dof_index, double> boundary_values;

      for (std::vector<unsigned int>::iterator it = magnetismDirichletMarkers.begin(); it != magnetismDirichletMarkers.end(); ++it)
      {
        VectorTools::interpolate_boundary_values(dof_handler, *it, ZeroFunction<DIM>(DIM), boundary_values);
        MatrixTools::apply_boundary_values(boundary_values, system_matrix, present_solution, system_rhs);
      }
    }

    void Solver::solveAlgebraicSystem()
    {
      if (PRINT_ALGEBRA)
      {
        std::cout << "  Printing system... " << std::endl;

        std::string matrix_file = "MagMatrix_";
        std::string rhs_file = "MagRhs_";

        std::ofstream matrix_out(matrix_file);
        std::ofstream rhs_out(rhs_file);

        system_matrix.print(matrix_out);
        system_rhs.print(rhs_out, 3, true, false);

        matrix_out.close();
        rhs_out.close();
      }

      std::cout << "Solving Mag..." << std::endl;

      direct_CustomSolver.initialize(system_matrix);

      direct_CustomSolver.vmult(present_solution, system_rhs);

      hanging_node_constraints.distribute(present_solution);
    }
  }
#pragma endregion

#pragma region assembling - flow
  namespace Flow
  {
    void Solver::localAssembleSystem(const dealii::hp::DoFHandler<DIM>::active_cell_iterator &cell, AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data)
    {
      scratch_data.hp_fe_values_Flow.reinit(cell);
      const dealii::FEValues<DIM> &fe_values_Flow = scratch_data.hp_fe_values_Flow.get_present_fe_values();

      const TYPENAME dealii::hp::DoFHandler<DIM>::active_cell_iterator &cellMag = GridTools::find_active_cell_around_point(this->magSolver->dof_handler, cell->center());
      scratch_data.hp_fe_values_Mag.reinit(cellMag);
      const dealii::FEValues<DIM> &fe_values_Mag = scratch_data.hp_fe_values_Mag.get_present_fe_values();

      const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      const unsigned int n_q_points = fe_values_Flow.n_quadrature_points;

      copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      copy_data.cell_matrix = 0;
      copy_data.cell_rhs.reinit(dofs_per_cell);
      copy_data.cell_rhs = 0;

      std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      std::vector<dealii::Vector<double> > prev_values_Flow(n_q_points, dealii::Vector<double>(DIM + 1));
      std::vector<std::vector<dealii::Tensor<1, DIM> > > prev_gradients_Flow(n_q_points, std::vector<dealii::Tensor<1, DIM> >(DIM + 1));

      std::vector<dealii::Vector<double> > prev_values_time_Flow(n_q_points, dealii::Vector<double>(DIM + 1));
      std::vector<std::vector<dealii::Tensor<1, DIM> > > prev_gradients_time_Flow(n_q_points, std::vector<dealii::Tensor<1, DIM> >(DIM + 1));

      std::vector<dealii::Vector<double> > prev_values_Mag(n_q_points, dealii::Vector<double>(DIM));
      std::vector<std::vector<dealii::Tensor<1, DIM> > > prev_gradients_Mag(n_q_points, std::vector<dealii::Tensor<1, DIM> >(DIM));

      std::vector<int> components(dofs_per_cell);

      fe_values_Flow.get_function_values(present_solution, prev_values_Flow);
      fe_values_Flow.get_function_gradients(present_solution, prev_gradients_Flow);

      fe_values_Flow.get_function_values(this->previous_sln_Flow, prev_values_time_Flow);
      fe_values_Flow.get_function_gradients(this->previous_sln_Flow, prev_gradients_time_Flow);

      fe_values_Mag.get_function_values(this->previous_sln_Mag, prev_values_Mag);
      fe_values_Mag.get_function_gradients(this->previous_sln_Mag, prev_gradients_Mag);


      // curl A from the previous iteration.
      std::vector<dealii::Tensor<1, DIM> > C(n_q_points);
      for (int i = 0; i < n_q_points; i++)
        C[i] = curl(prev_gradients_Mag[i][0], prev_gradients_Mag[i][1], prev_gradients_Mag[i][2]);

      std::vector<std::vector<double> > shape_value(dofs_per_cell, std::vector<double>(n_q_points));
      std::vector<double> JxW(n_q_points);
      std::vector<std::vector<dealii::Tensor<1, DIM> > > shape_grad(dofs_per_cell, std::vector<dealii::Tensor<1, DIM> >(n_q_points));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        components[i] = cell->get_fe().system_to_component_index(i).first;
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          shape_value[i][q_point] = fe_values_Flow.shape_value(i, q_point);
          shape_grad[i][q_point] = fe_values_Flow.shape_grad(i, q_point);
        }
      }
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        JxW[q_point] = fe_values_Flow.JxW(q_point);

      // Volumetric marker
      unsigned int marker = cell->material_id();

      // Geometrical points
      std::vector<dealii::Point<DIM> > points;
      points.reserve(dealii::DoFTools::max_dofs_per_face(dof_handler));
      points = fe_values_Flow.get_quadrature_points();

      // distribute local to global matrix
      copy_data.local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(copy_data.local_dof_indices);
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        dealii::Point<DIM> q_p = points[q_point];

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            // Velocity forms
            if (components[i] < DIM && components[j] < DIM)
            {
              // Coinciding indices.
              if (components[i] == components[j])
              {
                // Time derivative
                copy_data.cell_matrix(i, j) += shape_value[i][q_point]
                  * shape_value[j][q_point]
                  * JxW[q_point]
                  / TIME_STEP;

                // Diffusion
                copy_data.cell_matrix(i, j) += shape_grad[i][q_point]
                  * shape_grad[j][q_point]
                  * JxW[q_point]
                  / REYNOLDS;

                // Advection - 1/2
                copy_data.cell_matrix(i, j) += (prev_values_Flow[q_point][0] * shape_grad[j][q_point][0] + prev_values_Flow[q_point][1] * shape_grad[j][q_point][1] + prev_values_Flow[q_point][2] * shape_grad[j][q_point][2])
                  * shape_value[i][q_point]
                  * JxW[q_point];

                // Advection - 2/2
                copy_data.cell_matrix(i, j) += prev_gradients_Flow[q_point][components[i]][components[i]]
                  * shape_value[i][q_point]
                  * shape_value[j][q_point]
                  * JxW[q_point];

#pragma region NO_MOVEMENT_INDUCED_FORCE
                if (!NO_MOVEMENT_INDUCED_FORCE && marker == MARKER_FLUID)
                {
                  // sigma (u x B) x B WRT VELOCITIES - coinciding indices
                  for (int other_component = 0; other_component < DIM; other_component++)
                  {
                    if (other_component != components[i])
                    {
                      copy_data.cell_matrix(i, j) += SIGMA * C[q_point][other_component] * C[q_point][other_component]
                        * shape_value[i][q_point]
                        * shape_value[j][q_point]
                        * JxW[q_point]
                        / DENSITY;
                    }
                  }
                }
#pragma endregion
              }
              // NON-Coinciding indices.
              else
              {
                // Nonsymmetrical terms from N-S equations
                copy_data.cell_matrix(i, j) += prev_gradients_Flow[q_point][components[i]][components[j]]
                  * shape_value[i][q_point]
                  * shape_value[j][q_point]
                  * JxW[q_point];

#pragma region NO_MOVEMENT_INDUCED_FORCE
                if (!NO_MOVEMENT_INDUCED_FORCE && marker == MARKER_FLUID)
                {
                  // sigma (u x B) x B WRT VELOCITIES - NON-coinciding indices
                  copy_data.cell_matrix(i, j) -= SIGMA * C[q_point][components[i]] * C[q_point][components[j]]
                    * shape_value[i][q_point]
                    * shape_value[j][q_point]
                    * JxW[q_point]
                    / DENSITY;
                }
#pragma endregion
              }
            }

#pragma region PRESSURE
            // Pressure forms
            if (components[i] == DIM || components[j] == DIM)
            {
              // First let us do the last pseudo-row.
              // TODO
              // This is just anti-symmetry => optimize
              if (components[i] == DIM && components[j] < DIM)
              {
                copy_data.cell_matrix(i, j) += shape_value[i][q_point]
                  * shape_grad[j][q_point][components[j]]
                  * JxW[q_point];
              }
              else if (components[j] == DIM && components[i] < DIM)
              {
                copy_data.cell_matrix(i, j) -= shape_value[j][q_point]
                  * shape_grad[i][q_point][components[i]]
                  * JxW[q_point];
              }
            }
#pragma endregion
          }

          // Velocity rhs
          if (components[i] < DIM)
          {
            // Time derivative
            copy_data.cell_rhs(i) += shape_value[i][q_point]
              * (prev_values_Flow[q_point][components[i]] - prev_values_time_Flow[q_point][components[i]])
              * JxW[q_point]
              / TIME_STEP;

            // Diffusion
            copy_data.cell_rhs(i) += shape_grad[i][q_point]
              * prev_gradients_Flow[q_point][components[i]]
              * JxW[q_point]
              / REYNOLDS;

            // Pressure form
            copy_data.cell_rhs(i) -= shape_grad[i][q_point][components[i]]
              * prev_values_Flow[q_point][DIM]
              * JxW[q_point];

            copy_data.cell_rhs(i) += shape_value[i][q_point]
              * (prev_gradients_Flow[q_point][components[i]][0] * prev_values_Flow[q_point][0] + prev_gradients_Flow[q_point][components[i]][1] * prev_values_Flow[q_point][1] + prev_gradients_Flow[q_point][components[i]][2] * prev_values_Flow[q_point][2])
              * JxW[q_point];

#pragma region NO_MOVEMENT_INDUCED_FORCE
            if (!NO_MOVEMENT_INDUCED_FORCE && marker == MARKER_FLUID)
            {
              // Forces from magnetic field
              for (unsigned int j = 0; j < DIM; ++j)
              {
                if (j == components[i])
                {
                  for (int other_component = 0; other_component < DIM; other_component++)
                  {
                    if (other_component != components[i])
                    {
                      copy_data.cell_rhs(i) += SIGMA * C[q_point][other_component] * C[q_point][other_component]
                        * shape_value[i][q_point]
                        * prev_values_Flow[q_point][j]
                        * JxW[q_point]
                        / DENSITY;
                    }
                  }
                }
                else
                {
                  copy_data.cell_rhs(i) -= SIGMA * C[q_point][components[i]] * C[q_point][components[j]]
                    * shape_value[i][q_point]
                    * prev_values_Flow[q_point][j]
                    * JxW[q_point]
                    / DENSITY;
                }
              }
            }
#pragma endregion

#pragma region NO_EXT_CURR_DENSITY_FORCE
            if (!NO_EXT_CURR_DENSITY_FORCE && marker == MARKER_FLUID)
            {
              for (unsigned int j = 0; j < DIM; ++j)
              {
                if (components[i] == j)
                {
                  for (int other_component = 0; other_component < DIM; other_component++)
                  {
                    if (other_component != components[i])
                    {
                      double val = J_EXT(marker, other_component, q_p)
                        * shape_value[i][q_point]
                        * prev_gradients_Mag[q_point][j][other_component]
                        * JxW[q_point]
                        / DENSITY;

                      copy_data.cell_rhs(i) += val;
                    }
                  }
                }
                // - second part (NON-coinciding indices)
                else
                {
                  double val = J_EXT(marker, j, q_p)
                    * shape_value[i][q_point]
                    * prev_gradients_Mag[q_point][j][components[i]]
                    * JxW[q_point]
                    / DENSITY;
                  copy_data.cell_rhs(i) -= val;
                }
              }
            }
#pragma endregion
          }
#pragma region PRESSURE
          // Pressure rhs
          if (components[i] == DIM)
          {
            for (int vel_i = 0; vel_i < DIM; vel_i++)
            {
              copy_data.cell_rhs(i) += shape_value[i][q_point]
                * prev_gradients_Flow[q_point][vel_i][vel_i]
                * JxW[q_point];
            }
          }
#pragma endregion
        }
      }

      // distribute local to global matrix
      copy_data.local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(copy_data.local_dof_indices);

      copy_data.isAssembled = true;
    }

    void Solver::assemble_system()
    {
      system_matrix = 0;
      system_rhs = 0;

      dealii::hp::FEValues<DIM> hp_fe_values(feCollection, qCollection, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

      dealii::hp::DoFHandler<DIM>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

      dealii::WorkStream::run(cell, endc,
        *this,
        &Solver::localAssembleSystem,
        &Solver::copyLocalToGlobal,
        AssemblyScratchData(this->feCollection, this->mappingCollection, this->qCollection, this->magSolver->feCollection, this->magSolver->mappingCollection, this->magSolver->qCollection),
        AssemblyCopyData());

      this->finishAssembling();
    }

    void Solver::copyLocalToGlobal(const AssemblyCopyData &copy_data)
    {
      if (copy_data.isAssembled)
      {
        hanging_node_constraints.distribute_local_to_global(copy_data.cell_matrix,
          copy_data.cell_rhs,
          copy_data.local_dof_indices,
          system_matrix,
          system_rhs);
      }
    }

    void Solver::finishAssembling()
    {
      // Finally, we remove hanging nodes from the system and apply zero
      // boundary values to the linear system that defines the Newton updates
      // $\delta u^n$:
      hanging_node_constraints.condense(system_matrix);
      hanging_node_constraints.condense(system_rhs);

      ComponentMask velocity_mask(DIM + 1, false);
      for (int i = 0; i < DIM; i++)
        velocity_mask.set(i, true);

      std::map<types::global_dof_index, double> boundary_values;

      // Velocity
      for (std::vector<unsigned int>::iterator it = velocityDirichletMarkers.begin(); it != velocityDirichletMarkers.end(); ++it)
      {
        VectorTools::interpolate_boundary_values(dof_handler, *it, ZeroFunction<DIM>(DIM + 1), boundary_values, velocity_mask);
        MatrixTools::apply_boundary_values(boundary_values, system_matrix, newton_update, system_rhs);
      }
      if (INLET_VELOCITY_FIXED)
      {
        VectorTools::interpolate_boundary_values(dof_handler, INLET_VELOCITY_FIXED_BOUNDARY, ZeroFunction<DIM>(DIM + 1), boundary_values, velocity_mask);
        MatrixTools::apply_boundary_values(boundary_values, system_matrix, newton_update, system_rhs);
      }
    }

    void Solver::solveAlgebraicSystem(int inner_iteration)
    {
      if (PRINT_ALGEBRA)
      {
        std::cout << "  Printing system " << inner_iteration << "... " << std::endl;

        std::string matrix_file = "FlowMatrix_";
        std::string rhs_file = "FlowRhs_";

        matrix_file.append(std::to_string(inner_iteration));
        rhs_file.append(std::to_string(inner_iteration));

        std::ofstream matrix_out(matrix_file);
        std::ofstream rhs_out(rhs_file);

        system_matrix.print(matrix_out);
        system_rhs.print(rhs_out, 3, true, false);

        matrix_out.close();
        rhs_out.close();
      }

      std::cout << "Solving Flow..." << std::endl;

      direct_CustomSolver.initialize(system_matrix);

      // RHS for Newton is -F
      system_rhs *= -1.;
      direct_CustomSolver.vmult(newton_update, system_rhs);

      hanging_node_constraints.distribute(newton_update);

      present_solution.add(NEWTON_DAMPING, newton_update);
    }
  }
#pragma endregion

#pragma region postprocessor
  namespace Flow
  {
    class Postprocessor : public DataPostprocessor < DIM >
    {
    public:
      Postprocessor();
      virtual void compute_derived_quantities_vector(const std::vector<Vector<double> > &uh, const std::vector<std::vector<Tensor<1, DIM> > > &duh, const std::vector<std::vector<Tensor<2, DIM> > > &dduh, const std::vector<Point<DIM> > &normals, const std::vector<Point<DIM> >                  &evaluation_points, const dealii::types::material_id mat_id, std::vector<Vector<double> >                    &computed_quantities) const;
      virtual std::vector<std::string> get_names() const;
      virtual std::vector < DataComponentInterpretation::DataComponentInterpretation > get_data_component_interpretation() const;
      virtual UpdateFlags get_needed_update_flags() const;
    };

    Postprocessor::Postprocessor() : DataPostprocessor<DIM>()
    {}

    void Postprocessor::compute_derived_quantities_vector(const std::vector<Vector<double> > &uh, const std::vector<std::vector<Tensor<1, DIM> > > &duh, const std::vector<std::vector<Tensor<2, DIM> > > &dduh, const std::vector<Point<DIM> > &normals, const std::vector<Point<DIM> > &evaluation_points, const dealii::types::material_id mat_id, std::vector<Vector<double> > &computed_quantities) const
    {
      const unsigned int n_quadrature_points = uh.size();

      for (unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        // Velocities
        Tensor<1, DIM> v({ uh[q](0), uh[q](1), uh[q](2) });
        for (unsigned int d = 0; d < DIM; ++d)
          computed_quantities[q](d) = v[d];
        // Divergence
        computed_quantities[q](DIM) = duh[q][0][0] + duh[q][1][1] + duh[q][2][2];
        // Pressure
        computed_quantities[q](DIM + 1) = uh[q](DIM);
        // Material
        computed_quantities[q](DIM + 2) = mat_id;
      }
    }

    std::vector<std::string> Postprocessor::get_names() const
    {
      std::vector<std::string> names;
      for (unsigned int d = 0; d < DIM; ++d)
        names.push_back("velocity");
      names.push_back("div_velocity");
      names.push_back("pressure");
      names.push_back("material");
      return names;
    }

    std::vector<DataComponentInterpretation::DataComponentInterpretation> Postprocessor::get_data_component_interpretation() const
    {
      std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
      for (unsigned int d = 0; d < DIM; ++d)
        interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      return interpretation;
    }

    UpdateFlags Postprocessor::get_needed_update_flags() const
    {
      return update_values | update_gradients | update_quadrature_points;
    }

    void Solver::output_results(int time_iteration, int inner_iteration)
    {
      Postprocessor postprocessor;
      DataOut<DIM, hp::DoFHandler<DIM> > data_out;
      data_out.attach_dof_handler(dof_handler);
      const DataOut<DIM, hp::DoFHandler<DIM> >::DataVectorType data_vector_type = DataOut<DIM, hp::DoFHandler<DIM> >::type_dof_data;
      data_out.add_data_vector(present_solution, postprocessor);
      data_out.build_patches();
      std::string filename = "solutionFlow-";
      filename.append(std::to_string(time_iteration));
      filename.append("-");
      filename.append(std::to_string(inner_iteration));
      filename.append(".vtk");
      std::ofstream output(filename.c_str());
      data_out.write_vtk(output);
    }
  }

  namespace Mag
  {
    class Postprocessor : public DataPostprocessor < DIM >
    {
    public:
      Postprocessor();
      virtual void compute_derived_quantities_vector(const std::vector<Vector<double> > &uh, const std::vector<std::vector<Tensor<1, DIM> > > &duh, const std::vector<std::vector<Tensor<2, DIM> > > &dduh, const std::vector<Point<DIM> >                  &normals, const std::vector<Point<DIM> >                  &evaluation_points, const dealii::types::material_id mat_id, std::vector<Vector<double> >                    &computed_quantities) const;
      virtual std::vector<std::string> get_names() const;
      virtual std::vector < DataComponentInterpretation::DataComponentInterpretation > get_data_component_interpretation() const;
      virtual UpdateFlags get_needed_update_flags() const;
    };

    Postprocessor::Postprocessor() : DataPostprocessor<DIM>()
    {}

    void Postprocessor::compute_derived_quantities_vector(const std::vector<Vector<double> > &uh, const std::vector<std::vector<Tensor<1, DIM> > > &duh, const std::vector<std::vector<Tensor<2, DIM> > > &dduh, const std::vector<Point<DIM> > &normals, const std::vector<Point<DIM> > &evaluation_points, const dealii::types::material_id mat_id, std::vector<Vector<double> > &computed_quantities) const
    {
      const unsigned int n_quadrature_points = uh.size();

      for (unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        int index = 0;
        // A
        for (; index < DIM; index++)
          computed_quantities[q](index) = uh[q](index);
        // Curl A
        Tensor<1, DIM> A_x = duh[q][0];
        Tensor<1, DIM> A_y = duh[q][1];
        Tensor<1, DIM> A_z = duh[q][2];

        Tensor<1, DIM> B = curl(A_x, A_y, A_z);
        computed_quantities[q](index++) = B[0];
        computed_quantities[q](index++) = B[1];
        computed_quantities[q](index++) = B[2];

        Tensor<1, DIM> J_ext = Tensor<1, DIM>({ J_EXT(mat_id, 0, evaluation_points[q]), J_EXT(mat_id, 1, evaluation_points[q]), J_EXT(mat_id, 2, evaluation_points[q]) });
        Tensor<1, DIM> J_ext_xB = custom_cross_product(J_ext, B);
        computed_quantities[q](index++) = J_ext_xB[0];
        computed_quantities[q](index++) = J_ext_xB[1];
        computed_quantities[q](index++) = J_ext_xB[2];

        computed_quantities[q](index++) = J_ext[0];
        computed_quantities[q](index++) = J_ext[1];
        computed_quantities[q](index++) = J_ext[2];

        // Material id
        computed_quantities[q](index++) = mat_id;
      }
    }

    std::vector<std::string> Postprocessor::get_names() const
    {
      std::vector<std::string> names;
      for (unsigned int d = 0; d < DIM; ++d)
        names.push_back("A");
      for (unsigned int d = 0; d < DIM; ++d)
        names.push_back("B");
      for (unsigned int d = 0; d < DIM; ++d)
        names.push_back("(J_ext)xB");
      for (unsigned int d = 0; d < DIM; ++d)
        names.push_back("J_ext");
      names.push_back("material");
      return names;
    }

    std::vector<DataComponentInterpretation::DataComponentInterpretation> Postprocessor::get_data_component_interpretation() const
    {
      std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
      for (unsigned int d = 0; d < DIM; ++d)
        interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
      for (unsigned int d = 0; d < DIM; ++d)
        interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
      for (unsigned int d = 0; d < DIM; ++d)
        interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
      for (unsigned int d = 0; d < DIM; ++d)
        interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

      interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      return interpretation;
    }

    UpdateFlags Postprocessor::get_needed_update_flags() const
    {
      return update_values | update_gradients | update_quadrature_points;
    }

    void Solver::output_results(int time_iteration)
    {
      Postprocessor postprocessor;
      DataOut<DIM, hp::DoFHandler<DIM> > data_out;
      data_out.attach_dof_handler(dof_handler);
      const DataOut<DIM, hp::DoFHandler<DIM> >::DataVectorType data_vector_type = DataOut<DIM, hp::DoFHandler<DIM> >::type_dof_data;
      data_out.add_data_vector(present_solution, postprocessor);
      data_out.build_patches();
      std::string filename = "solutionMag-";
      filename.append(std::to_string(time_iteration));
      filename.append(".vtk");
      std::ofstream output(filename.c_str());
      data_out.write_vtk(output);
    }
  }
#pragma endregion

#pragma region startup
  namespace Flow
  {
    void Solver::solveOneStep(Vector<double> previous_sln_Mag, int iteration)
    {
      this->previous_sln_Mag = previous_sln_Mag;

      this->setup_system(iteration == 0);

      this->assemble_system();

      double previous_res = 0.;


      for (unsigned int inner_iteration = 0; inner_iteration < NEWTON_ITERATIONS; ++inner_iteration)
      {
        std::cout << "Assembling Flow..." << std::endl;
        assemble_system();
        previous_res = system_rhs.l2_norm();
        std::cout << "  Residual: " << previous_res << std::endl;

        solveAlgebraicSystem(inner_iteration);

        if (PRINT_ALGEBRA)
        {
          std::cout << "  Printing solution " << inner_iteration << "... " << std::endl;

          std::string sln_file = "Sln_";
          sln_file.append(std::to_string(inner_iteration));
          std::ofstream sln_out(sln_file);
          newton_update.print(sln_out, 8, true, false);
          sln_out.close();
        }

        output_results(iteration, inner_iteration);

        if (previous_res < NEWTON_RESIDUAL_THRESHOLD)
          break;
      }
    }
  }

  namespace Mag
  {
    void Solver::solveOneStep(Vector<double> previous_sln_Flow, int iteration)
    {
      this->previous_sln_Flow = previous_sln_Flow;

      this->setup_system(iteration == 0);

      std::cout << "Assembling Mag..." << std::endl;
      this->assemble_system();

      solveAlgebraicSystem();

      if (PRINT_ALGEBRA)
      {
        std::cout << "  Printing solution... " << std::endl;

        std::string sln_file = "Sln_";
        std::ofstream sln_out(sln_file);
        this->present_solution.print(sln_out, 8, true, false);
        sln_out.close();
      }

      output_results(iteration);
    }
  }

  Solver::Solver()
  {

  }
  Solver::~Solver()
  {

  }
  void Solver::run()
  {
    this->flowSolver = new Flow::Solver();
    this->magSolver = new Mag::Solver();

    this->flowSolver->init_discretization(this->magSolver);
    this->magSolver->init_discretization(this->flowSolver);

    double time = 0.0;
    for (int iteration = 0; time < T_END; time += TIME_STEP, iteration++)
    {
      if(iteration == 0)
        this->magSolver->solveOneStep(this->flowSolver->present_solution, iteration);
      this->flowSolver->solveOneStep(this->magSolver->present_solution, iteration);
    }
  }
#pragma endregion
}


int main()
{
  try
  {
    using namespace dealii;
    deallog.depth_console(0);

    MHD::Solver fe_problem;
    fe_problem.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
      << "----------------------------------------------------"
      << std::endl;
    std::cerr << "Exception on processing: " << std::endl
      << exc.what() << std::endl
      << "Aborting!" << std::endl
      << "----------------------------------------------------"
      << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
      << "----------------------------------------------------"
      << std::endl;
    std::cerr << "Unknown exception!" << std::endl
      << "Aborting!" << std::endl
      << "----------------------------------------------------"
      << std::endl;
    return 1;
  }
  return 0;
}
