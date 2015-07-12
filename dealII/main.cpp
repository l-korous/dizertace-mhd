// A comes after v, p in the system (hardcoded)
// Assembling is done in the form J() | = F()
// i.e. the negative sign on the RHS is handled after assembling of F.

#define DIM 3
#define RANDOM_INITIAL_GUESS 0.0
#define DEPTH 5
#define MAGNET_SIZE 3
#define AIR_LAYER_THICKNESS 3
#define INIT_REF_NUM 14

#pragma region TESTING

const bool PRINT_ALGEBRA = false, PRINT_INIT_SLN = true;
const bool A_ONLY_LAPLACE = false;
const bool NO_MOVEMENT_INDUCED_FORCE = true;
const bool NO_EXT_CURR_DENSITY_FORCE = false;
const bool A_LINEAR_WRT_Y = true;

#pragma endregion

#include <deal.II/base/point.h>
#include <vector>
const dealii::Point<DIM> p1(0., 0., 0.);
const dealii::Point<DIM> p2(1., 1., 1.);
const dealii::Point<DIM> pc((p2(0) - p1(0)) / 2., (p2(1) - p1(1)) / 2., (p2(2) - p1(2)) / 2.);
const std::vector<unsigned int> refinements({ INIT_REF_NUM, INIT_REF_NUM, DEPTH });
const dealii::Point<DIM> singleLayerThickness((p2(0) - p1(0)) / ((double)refinements[0]), (p2(1) - p1(1)) / ((double)refinements[1]), (p2(2) - p1(2)) / ((double)refinements[2]));
const double flowChannelWidth = singleLayerThickness(0) * INIT_REF_NUM - MAGNET_SIZE - AIR_LAYER_THICKNESS;
const double flowChannelBoundaries[2] = { pc(0) - flowChannelWidth, pc(0) + flowChannelWidth };


// boundary id
const unsigned int BOUNDARY_FRONT = 1;
const unsigned int BOUNDARY_RIGHT = 2;
const unsigned int BOUNDARY_BACK = 3;
const unsigned int BOUNDARY_LEFT = 4;
const unsigned int BOUNDARY_BOTTOM = 5;
const unsigned int BOUNDARY_TOP = 6;
const unsigned int BOUNDARY_VEL_WALL = 7;
const unsigned int BOUNDARY_ELECTRODES = 8;

std::vector<unsigned int> velocityDirichletMarkers({ BOUNDARY_VEL_WALL, BOUNDARY_VEL_WALL });
std::vector<unsigned int> magnetismDirichletMarkers({ BOUNDARY_BOTTOM, BOUNDARY_TOP, BOUNDARY_LEFT, BOUNDARY_RIGHT });
std::vector<unsigned int> currentDirichletMarkers({ BOUNDARY_ELECTRODES, BOUNDARY_ELECTRODES });

const bool INLET_VELOCITY_FIXED = false;
const unsigned int INLET_VELOCITY_FIXED_BOUNDARY = BOUNDARY_BACK;
const double INLET_VELOCITY_AMPLITUDE = 10.0;

const unsigned int POLYNOMIAL_DEGREE_MAG = 2;
const unsigned int POLYNOMIAL_DEGREE_E = 2;

const double MU = 1.2566371e-6;
const double MU_R = 1.;

// material id 
const unsigned int MARKER_AIR = 0;
const unsigned int MARKER_MAGNET = 1;
const unsigned int MARKER_FLUID = 2;
const unsigned int MARKER_ELECTRODE = 3;

// These are according to components
const double B_R[3] = { 0., -1.e-4, 0. };
// These are according to material ids
const double SIGMA[3] = { 0., 0., 3.e6 };
const double J_EXT_VAL = 1.e-4;
double J_EXT(int marker, int component, dealii::Point<DIM> p)
{
  if (marker == MARKER_FLUID && component == 0)
    return J_EXT_VAL;
  return 0.;
}

const double REYNOLDS = 5.;

const double NEWTON_DAMPING = .7;
const int NEWTON_ITERATIONS = 100;
const double NEWTON_RESIDUAL_THRESHOLD = 1e-10;

const int COMPONENT_COUNT = 2 * DIM + 1;

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/grid/grid_out.h>
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

namespace Step15
{
  using namespace dealii;

  template <int dim>
  class AssemblyScratchData
  {
  public:
    AssemblyScratchData(const dealii::hp::FECollection<dim> &feCollection,
      const dealii::hp::MappingCollection<dim> &mappingCollection,
      const dealii::hp::QCollection<dim> &quadratureFormulas);
    AssemblyScratchData(const AssemblyScratchData &scratch_data);

    dealii::hp::FEValues<dim> hp_fe_values;
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

  template <int dim>
  class Postprocessor : public DataPostprocessor < dim >
  {
  public:
    Postprocessor();
    virtual
      void
      compute_derived_quantities_vector(const std::vector<Vector<double> > &uh,
      const std::vector<std::vector<Tensor<1, dim> > > &duh,
      const std::vector<std::vector<Tensor<2, dim> > > &dduh,
      const std::vector<Point<dim> >                  &normals,
      const std::vector<Point<dim> >                  &evaluation_points,
      const dealii::types::material_id mat_id,
      std::vector<Vector<double> >                    &computed_quantities) const;
    virtual std::vector<std::string> get_names() const;
    virtual std::vector < DataComponentInterpretation::DataComponentInterpretation > get_data_component_interpretation() const;
    virtual UpdateFlags get_needed_update_flags() const;
  };

  template <int dim>
  class CustomSolver
  {
  public:
    CustomSolver();
    ~CustomSolver();

    void run();

  private:
    void setup_system(const bool initial_step);
    void assemble_system();

    void init_discretization();
    void add_markers(typename Triangulation<dim>::cell_iterator cell);

    void output_results(int inner_iteration);

    void localAssembleSystem(const typename dealii::hp::DoFHandler<dim>::active_cell_iterator &iter,
      AssemblyScratchData<dim> &scratch_data,
      AssemblyCopyData &copy_data);

    void copyLocalToGlobal(const AssemblyCopyData &copy_data);

    void finishAssembling();

    void solveAlgebraicSystem(int inner_iteration);
    void refine_mesh();
    void set_boundary_values();

    Triangulation<dim>   triangulation;
    hp::DoFHandler<dim>      dof_handler;

    dealii::hp::FECollection<dim> feCollection;
    dealii::hp::MappingCollection<dim> mappingCollection;
    dealii::hp::QCollection<dim> qCollection;
    dealii::hp::QCollection<dim - 1> qCollectionFace;

    ConstraintMatrix     hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       present_solution;
    Vector<double>       newton_update;
    Vector<double>       system_rhs;

    dealii::SparseDirectUMFPACK direct_CustomSolver;
  };

#pragma region BC_values
  template <int dim>
  class BoundaryValuesInlet : public Function < dim >
  {
  public:
    // Function has to be called with the proper number of components
    BoundaryValuesInlet() : Function<dim>(COMPONENT_COUNT) {}

    virtual double value(const Point<dim>   &p, const unsigned int  component = 0) const;
  };

  template <>
  double BoundaryValuesInlet<3>::value(const Point<3> &p, const unsigned int component) const
  {
    /*
    if (component == 1)
    {
    double p_x[2] = { p1(0) + (AIR_LAYER_THICKNESS + COIL_LAYER_THICKNESS) * (p2(0) - p1(0)) / INIT_REF_NUM, p2(0) - (AIR_LAYER_THICKNESS + COIL_LAYER_THICKNESS) * (p2(0) - p1(0)) / INIT_REF_NUM };
    double p_z[2] = { p1(2) + AIR_LAYER_THICKNESS * (p2(2) - p1(2)) / INIT_REF_NUM, p2(2) - AIR_LAYER_THICKNESS * (p2(2) - p1(2)) / INIT_REF_NUM };
    return INLET_VELOCITY_AMPLITUDE * ((p(0) - p_x[0]) * (p_x[1] - p(0))) * ((p(2) - p_z[0]) * (p_z[1] - p(2)));
    }

    else
    */
    return 0;
  }

  template <int dim>
  class BoundaryValuesWall : public Function < dim >
  {
  public:
    // Function has to be called with the proper number of components
    BoundaryValuesWall() : Function<dim>(COMPONENT_COUNT) {}

    virtual double value(const Point<dim>   &p, const unsigned int  component = 0) const;
  };

  template <int dim>
  double BoundaryValuesWall<dim>::value(const Point<dim> &p, const unsigned int component) const
  {
    return 0.;
  }

  template <int dim>
  void CustomSolver<dim>::set_boundary_values()
  {
    std::map<types::global_dof_index, double> boundary_values;

    ComponentMask velocity_mask(COMPONENT_COUNT, false);
    for (int i = 0; i < dim; i++)
      velocity_mask.set(i, true);

    ComponentMask magnetic_field_mask(COMPONENT_COUNT, false);
    for (int i = dim + 1; i < COMPONENT_COUNT; i++)
      magnetic_field_mask.set(i, true);

    for (int i = 0; i < this->dof_handler.n_dofs(); i++)
      present_solution(i) = RANDOM_INITIAL_GUESS;

    // Velocity
    for (std::vector<unsigned int>::iterator it = velocityDirichletMarkers.begin(); it != velocityDirichletMarkers.end(); ++it)
    {
      VectorTools::interpolate_boundary_values(dof_handler, *it, BoundaryValuesWall<dim>(), boundary_values, velocity_mask);
      for (std::map<types::global_dof_index, double>::const_iterator p = boundary_values.begin(); p != boundary_values.end(); ++p)
        present_solution(p->first) = p->second;
    }
    if (INLET_VELOCITY_FIXED)
    {
      VectorTools::interpolate_boundary_values(dof_handler, BOUNDARY_BOTTOM, BoundaryValuesInlet<dim>(), boundary_values, velocity_mask);
      for (std::map<types::global_dof_index, double>::const_iterator p = boundary_values.begin(); p != boundary_values.end(); ++p)
        present_solution(p->first) = p->second;
    }

    // Magnetism
    for (std::vector<unsigned int>::iterator it = magnetismDirichletMarkers.begin(); it != magnetismDirichletMarkers.end(); ++it)
    {
      VectorTools::interpolate_boundary_values(dof_handler, *it, BoundaryValuesWall<dim>(), boundary_values, magnetic_field_mask);
      for (std::map<types::global_dof_index, double>::const_iterator p = boundary_values.begin(); p != boundary_values.end(); ++p)
        present_solution(p->first) = p->second;
    }
  }

#pragma endregion

#pragma region Curl
  dealii::Tensor<1, 2> curl(dealii::Tensor<1, 2>& gradient)
  {
    dealii::Tensor<1, 2> result;
    throw "2D not implemented";
  }

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

#pragma region Postprocessor

  template <int dim>
  Postprocessor<dim>::Postprocessor() : DataPostprocessor<dim>()
  {}

  template<int dim>
  void CustomSolver<dim>::output_results(int inner_iteration)
  {
    Postprocessor<dim> postprocessor;
    DataOut<dim, hp::DoFHandler<dim> > data_out;
    data_out.attach_dof_handler(dof_handler);
    const typename DataOut<dim, hp::DoFHandler<dim> >::DataVectorType data_vector_type = DataOut<dim, hp::DoFHandler<dim> >::type_dof_data;
    data_out.add_data_vector(present_solution, postprocessor);
    data_out.build_patches();
    std::string filename = "solution";
    filename.append(std::to_string(inner_iteration));
    filename.append(".vtk");
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
  }

  template <int dim>
  void
    Postprocessor<dim>::
    compute_derived_quantities_vector(const std::vector<Vector<double> >              &uh,
    const std::vector<std::vector<Tensor<1, dim> > > &duh,
    const std::vector<std::vector<Tensor<2, dim> > > &dduh,
    const std::vector<Point<dim> >                  &normals,
    const std::vector<Point<dim> >                  &evaluation_points,
    const dealii::types::material_id mat_id,
    std::vector<Vector<double> >                    &computed_quantities) const
  {
    const unsigned int n_quadrature_points = uh.size();

    for (unsigned int q = 0; q < n_quadrature_points; ++q)
    {
      // Velocities
      Tensor<1, dim> v({ uh[q](0), uh[q](1), uh[q](2) });
      for (unsigned int d = 0; d < dim; ++d)
        computed_quantities[q](d) = v[d];
      // Divergence
      computed_quantities[q](dim) = duh[q][0][0] + duh[q][1][1] + duh[q][2][2];
      // Pressure
      computed_quantities[q](dim + 1) = uh[q](dim);
      // A
      for (unsigned int d = dim + 2; d < (2 * dim) + 2; ++d)
        computed_quantities[q](d) = uh[q](d - 1);
      // Curl A
      Tensor<1, dim> A_x = duh[q][dim + 1];
      Tensor<1, dim> A_y = duh[q][dim + 2];
      Tensor<1, dim> A_z = duh[q][dim + 3];

      Tensor<1, dim> B = curl(A_x, A_y, A_z);
      computed_quantities[q](2 * dim + 2) = B[0];
      computed_quantities[q](2 * dim + 3) = B[1];
      computed_quantities[q](2 * dim + 4) = B[2];

      Tensor<1, dim> v_x_B = custom_cross_product(v, B);
      computed_quantities[q](3 * dim + 2) = v_x_B[0];
      computed_quantities[q](3 * dim + 3) = v_x_B[1];
      computed_quantities[q](3 * dim + 4) = v_x_B[2];

      Tensor<1, dim> v_x_B_xB = custom_cross_product(v_x_B, B);
      computed_quantities[q](4 * dim + 2) = v_x_B_xB[0];
      computed_quantities[q](4 * dim + 3) = v_x_B_xB[1];
      computed_quantities[q](4 * dim + 4) = v_x_B_xB[2];

      Tensor<1, dim> J_ext = Tensor<1, dim>({ J_EXT(mat_id, 0, evaluation_points[q]), J_EXT(mat_id, 1, evaluation_points[q]), J_EXT(mat_id, 2, evaluation_points[q]) });
      Tensor<1, dim> J_ext_xB = custom_cross_product(J_ext, B);
      computed_quantities[q](5 * dim + 2) = J_ext_xB[0];
      computed_quantities[q](5 * dim + 3) = J_ext_xB[1];
      computed_quantities[q](5 * dim + 4) = J_ext_xB[2];

      computed_quantities[q](6 * dim + 2) = J_ext[0];
      computed_quantities[q](6 * dim + 3) = J_ext[1];
      computed_quantities[q](6 * dim + 4) = J_ext[2];

      // Velocity divergence
      computed_quantities[q](6 * dim + 5) = mat_id;
    }
  }

  template <int dim>
  std::vector<std::string>
    Postprocessor<dim>::
    get_names() const
  {
    std::vector<std::string> names;
    for (unsigned int d = 0; d < dim; ++d)
      names.push_back("velocity");
    names.push_back("div_velocity");
    names.push_back("pressure");
    for (unsigned int d = 0; d < dim; ++d)
      names.push_back("A");
    for (unsigned int d = 0; d < dim; ++d)
      names.push_back("B");
    for (unsigned int d = 0; d < dim; ++d)
      names.push_back("vxB");
    for (unsigned int d = 0; d < dim; ++d)
      names.push_back("(vxB)xB");
    for (unsigned int d = 0; d < dim; ++d)
      names.push_back("(J_ext)xB");
    for (unsigned int d = 0; d < dim; ++d)
      names.push_back("J_ext");
    names.push_back("material");
    //names.push_back("J_ext_curl_A");
    //names.push_back("J_ind_curl_A");
    return names;
  }


  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    Postprocessor<dim>::
    get_data_component_interpretation() const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    return interpretation;
  }

  template <int dim>
  UpdateFlags Postprocessor<dim>::
    get_needed_update_flags() const
  {
    return update_values | update_gradients | update_quadrature_points;
  }

#pragma endregion

  template <int dim>
  CustomSolver<dim>::CustomSolver() : dof_handler(triangulation)
  {
    // The first (default) FE system - identified by cell->active_fe_index is for the entire system, i.e. subdomains where both equations for fluid and for magnetism are solved
    {
      std::vector<const dealii::FiniteElement<dim> *> fes;
      std::vector<unsigned int> multiplicities;

      // Velocity
      fes.push_back(new dealii::FE_Q<dim>(2));
      multiplicities.push_back(dim);

      // Pressure
      fes.push_back(new dealii::FE_Q<dim>(1));
      multiplicities.push_back(1);

      // A
      fes.push_back(new dealii::FE_Q<dim>(POLYNOMIAL_DEGREE_MAG));
      multiplicities.push_back(dim);
      feCollection.push_back(dealii::FESystem<dim, dim>(fes, multiplicities));
    }

    // Second is only magnetism (coils + air)
    {
      std::vector<const dealii::FiniteElement<dim> *> fes;
      std::vector<unsigned int> multiplicities;

      // Velocity - MISSING
      fes.push_back(new dealii::FE_Nothing<dim>());
      multiplicities.push_back(dim);

      // Pressure - MISSING
      fes.push_back(new dealii::FE_Nothing<dim>());
      multiplicities.push_back(1);

      // A
      fes.push_back(new dealii::FE_Q<dim>(POLYNOMIAL_DEGREE_MAG));
      multiplicities.push_back(dim);
      feCollection.push_back(dealii::FESystem<dim, dim>(fes, multiplicities));
    }

    mappingCollection.push_back(dealii::MappingQ<dim>(1, true));

    // TODO
    // optimize, but the problem is the most consuming product is 2 * value, 1 * derivative which is basically this.
    qCollection.push_back(dealii::QGauss<dim>(3 * std::max<unsigned int>(POLYNOMIAL_DEGREE_MAG, 2)));
    qCollectionFace.push_back(dealii::QGauss<dim - 1>(3 * std::max<unsigned int>(POLYNOMIAL_DEGREE_MAG, 2)));
  }

  template <int dim>
  CustomSolver<dim>::~CustomSolver()
  {
    dof_handler.clear();
  }

  template <int dim>
  void CustomSolver<dim>::setup_system(const bool initial_step)
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

    // The remaining parts of the function are the same as in step-6.
    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, c_sparsity);
    hanging_node_constraints.condense(c_sparsity);
    sparsity_pattern.copy_from(c_sparsity);
    system_matrix.reinit(sparsity_pattern);

    std::cout << "System set up, " << triangulation.n_active_cells() << " cells, " << dof_handler.n_dofs() << " dofs" << std::endl;
  }

  template <int dim>
  void CustomSolver<dim>::assemble_system()
  {
    system_matrix = 0;
    system_rhs = 0;

    dealii::hp::FEValues<dim> hp_fe_values(feCollection, qCollection, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    typename dealii::hp::DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

    dealii::WorkStream::run(cell, endc,
      *this,
      &CustomSolver<dim>::localAssembleSystem,
      &CustomSolver<dim>::copyLocalToGlobal,
      AssemblyScratchData<dim>(this->feCollection, this->mappingCollection, this->qCollection),
      AssemblyCopyData());

    this->finishAssembling();
  }


  template <int dim>
  AssemblyScratchData<dim>::AssemblyScratchData(const dealii::hp::FECollection<dim> &feCollection,
    const dealii::hp::MappingCollection<dim> &mappingCollection,
    const dealii::hp::QCollection<dim> &quadratureFormulas)
    :
    hp_fe_values(mappingCollection, feCollection, quadratureFormulas,
    dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values)
  {}

  template <int dim>
  AssemblyScratchData<dim>::AssemblyScratchData(const AssemblyScratchData &scratch_data)
    :
    hp_fe_values(scratch_data.hp_fe_values.get_mapping_collection(),
    scratch_data.hp_fe_values.get_fe_collection(),
    scratch_data.hp_fe_values.get_quadrature_collection(),
    dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values)
  {}

  AssemblyCopyData::AssemblyCopyData()
    : isAssembled(false), cell_matrix(0), cell_rhs(0)
  {}

  template <int dim>
  void CustomSolver<dim>::localAssembleSystem(const typename dealii::hp::DoFHandler<dim>::active_cell_iterator &cell,
    AssemblyScratchData<dim> &scratch_data,
    AssemblyCopyData &copy_data)
  {
    // std::cout << " assembling cell number " << cell_number++ << std::endl;

    scratch_data.hp_fe_values.reinit(cell);
    const dealii::FEValues<dim> &fe_values = scratch_data.hp_fe_values.get_present_fe_values();
    const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
    const unsigned int n_q_points = fe_values.n_quadrature_points;

    copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    copy_data.cell_matrix = 0;
    copy_data.cell_rhs.reinit(dofs_per_cell);
    copy_data.cell_rhs = 0;

    std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);

    std::vector<dealii::Tensor<1, dim> > v_prev(n_q_points, dealii::Tensor<1, dim>(dim));
    std::vector<dealii::Vector<double> > old_solution_values(n_q_points, dealii::Vector<double>(COMPONENT_COUNT));
    std::vector<std::vector<dealii::Tensor<1, dim> > > old_solution_gradients(n_q_points, std::vector<dealii::Tensor<1, dim> >(COMPONENT_COUNT));
    std::vector<std::vector<dealii::Tensor<1, dim> > > A_prev_gradients(n_q_points, std::vector<dealii::Tensor<1, dim> >(dim));

    std::vector<int> components(dofs_per_cell);

    fe_values.get_function_values(present_solution, old_solution_values);
    fe_values.get_function_gradients(present_solution, old_solution_gradients);

    // Only velocity values for simpler form expressions
    for (int i = 0; i < old_solution_values.size(); i++)
    {
      for (int j = 0; j < dim; j++)
        v_prev[i][j] = old_solution_values[i][j];
    }
    // Only magnetism gradients for simpler form expressions
    for (int i = 0; i < old_solution_gradients.size(); i++)
    {
      for (int j = 0; j < dim; j++)
        A_prev_gradients[i][j] = old_solution_gradients[i][j + dim + 1];
    }

    // curl A from the previous iteration.
    std::vector<dealii::Tensor<1, dim> > C(n_q_points);
    for (int i = 0; i < n_q_points; i++)
      C[i] = curl(old_solution_gradients[i][dim + 1], old_solution_gradients[i][dim + 2], old_solution_gradients[i][dim + 3]);

    std::vector<std::vector<double> > shape_value(dofs_per_cell, std::vector<double>(n_q_points));
    std::vector<double> JxW(n_q_points);
    std::vector<std::vector<dealii::Tensor<1, dim> > > shape_grad(dofs_per_cell, std::vector<dealii::Tensor<1, dim> >(n_q_points));

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      components[i] = cell->get_fe().system_to_component_index(i).first;
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        shape_value[i][q_point] = fe_values.shape_value(i, q_point);
        shape_grad[i][q_point] = fe_values.shape_grad(i, q_point);
      }
    }
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      JxW[q_point] = fe_values.JxW(q_point);

    // Volumetric marker
    unsigned int marker = cell->material_id();

    // Geometrical points
    std::vector<dealii::Point<dim> > points;
    points.reserve(dealii::DoFTools::max_dofs_per_face(dof_handler));
    points = fe_values.get_quadrature_points();

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
      dealii::Point<dim> q_p = points[q_point];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        int components_mag_i = components[i] - dim - 1;
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          int components_mag_j = components[j] - dim - 1;

          // Velocity forms
          if (components[i] < dim && components[j] < dim)
          {
            // Coinciding indices.
            if (components[i] == components[j])
            {
              // Diffusion
              copy_data.cell_matrix(i, j) += shape_grad[i][q_point]
                * shape_grad[j][q_point]
                * JxW[q_point]
                / REYNOLDS;

              // Advection - 1/2
              copy_data.cell_matrix(i, j) += v_prev[q_point]
                * shape_grad[j][q_point]
                * shape_value[i][q_point]
                * JxW[q_point];

              // Advection - 2/2
              copy_data.cell_matrix(i, j) += old_solution_gradients[q_point][components[i]][components[i]]
                * shape_value[i][q_point]
                * shape_value[j][q_point]
                * JxW[q_point];

#pragma region NO_MOVEMENT_INDUCED_FORCE
              if (!NO_MOVEMENT_INDUCED_FORCE && marker == MARKER_FLUID)
              {
                // sigma (u x B) x B WRT VELOCITIES - coinciding indices
                for (int other_component = 0; other_component < dim; other_component++)
                {
                  if (other_component != components[i])
                  {
                    copy_data.cell_matrix(i, j) += SIGMA[marker] * C[q_point][other_component] * C[q_point][other_component]
                      * shape_value[i][q_point]
                      * shape_value[j][q_point]
                      * JxW[q_point];
                  }
                }
              }
#pragma endregion
            }
            // NON-Coinciding indices.
            else
            {
              // Nonsymmetrical terms from N-S equations
              copy_data.cell_matrix(i, j) += old_solution_gradients[q_point][components[i]][components[j]]
                * shape_value[i][q_point]
                * shape_value[j][q_point]
                * JxW[q_point];

#pragma region NO_MOVEMENT_INDUCED_FORCE
              if (!NO_MOVEMENT_INDUCED_FORCE && marker == MARKER_FLUID)
              {
                // sigma (u x B) x B WRT VELOCITIES - NON-coinciding indices
                copy_data.cell_matrix(i, j) -= SIGMA[marker] * C[q_point][components[i]] * C[q_point][components[j]]
                  * shape_value[i][q_point]
                  * shape_value[j][q_point]
                  * JxW[q_point];
              }
#pragma endregion
            }
          }

          // [J_{ext} + Sigma (u x B)] x B WRT MAGNETISM
          if (components[i] < dim && components[j] > dim)
          {
#pragma region NO_MOVEMENT_INDUCED_FORCE
            if (!NO_MOVEMENT_INDUCED_FORCE && marker == MARKER_FLUID)
            {
#pragma region paperToCodeHelpers
#define v_x v_prev[q_point][0]
#define v_y v_prev[q_point][1]
#define v_z v_prev[q_point][2]

              // Previous derivatives - we never need An_k_k
#define An_x_y old_solution_gradients[q_point][4][1]
#define An_x_z old_solution_gradients[q_point][4][2]

#define An_y_x old_solution_gradients[q_point][5][0]
#define An_y_z old_solution_gradients[q_point][5][2]

#define An_z_x old_solution_gradients[q_point][6][0]
#define An_z_y old_solution_gradients[q_point][6][1]

              // Previous curl
#define C_x C[q_point][0]
#define C_y C[q_point][1]
#define C_z C[q_point][2]

              // Current A - ith component (component the derivative is with respect to).
#define Ai_x shape_grad[j][q_point][0]
#define Ai_y shape_grad[j][q_point][1]
#define Ai_z shape_grad[j][q_point][2]
#pragma endregion
              double value = 0.;
              switch (components[i])
              {
              case 0:
                switch (components[j] - dim - 1)
                {
                case 0:
                  value = (v_z * C_x * Ai_y) - (v_y * C_x * Ai_z)
                    - (v_x * 2. * C_z * Ai_y) + (v_x * 2. * C_y * Ai_z);
                  break;
                case 1:
                  value = (v_y * C_y * Ai_z)
                    + (v_x * 2. * C_z * Ai_x)
                    - (v_z * (C_x * Ai_x - C_z * Ai_z));
                  break;
                case 2:
                  value = -(v_z * C_z * Ai_y)
                    - (v_x * 2. * C_y * Ai_x)
                    - (v_y * (C_y * Ai_y - C_x * Ai_x));
                  break;
                }
                break;
                break;
              case 1:
                switch (components[j] - dim - 1)
                {
                case 0:
                  value = -(v_x * C_x * Ai_z)
                    - (v_y * 2. * C_z * Ai_y)
                    - (v_z * (C_z * Ai_z - C_y * Ai_y));
                  break;
                case 1:
                  value = (v_x * C_y * Ai_z) - (v_z * C_y * Ai_x)
                    - (v_y * 2. * C_x * Ai_z) + (v_y * 2.* C_z * Ai_x);
                  break;
                case 2:
                  value = (v_z * C_z * Ai_x)
                    + (v_y * 2. * C_x * Ai_y)
                    - (v_x * (C_y * Ai_y - C_x * Ai_x));
                  break;
                }
                break;
                break;
              case 2:
                switch (components[j] - dim - 1)
                {
                case 0:
                  value = (v_x * C_x * Ai_y)
                    + (v_z * 2. * C_y * Ai_z)
                    - (v_y * (C_z * Ai_z - C_y * Ai_y));
                  break;
                case 1:
                  value = -(v_y * C_y * Ai_x)
                    - (v_z * 2. * C_x * Ai_z)
                    - (v_x * (C_x * Ai_x - C_z * Ai_z));
                  break;
                case 2:
                  value = (v_y * C_z * Ai_x) - (v_x * C_z * Ai_y)
                    - (v_z * 2. * C_y * Ai_x) + (v_z * 2. * C_x * Ai_y);
                  break;
                }
                break;
              }
              copy_data.cell_matrix(i, j) += SIGMA[marker] * value
                * shape_value[i][q_point]
                * JxW[q_point];
            }
#pragma endregion

#pragma region NO_EXT_CURR_DENSITY_FORCE
            if (!NO_EXT_CURR_DENSITY_FORCE && marker == MARKER_FLUID)
            {
              // (J_ext x (\Nabla x A))
              // - first part (coinciding indices)
              if (components[i] == components_mag_j)
              {
                for (int other_component = 0; other_component < dim; other_component++)
                {
                  if (other_component != components[i])
                  {
                    copy_data.cell_matrix(i, j) += J_EXT(marker, other_component, q_p)
                      * shape_value[i][q_point]
                      * shape_grad[j][q_point][other_component]
                      * JxW[q_point];
                  }
                }
              }
              // - second part (NON-coinciding indices)
              else
              {
                copy_data.cell_matrix(i, j) -= J_EXT(marker, components_mag_j, q_p)
                  * shape_value[i][q_point]
                  * shape_grad[j][q_point][components[i]]
                  * JxW[q_point];
              }
            }
#pragma endregion
          }

#pragma region PRESSURE
          // Pressure forms
          if (components[i] == dim || components[j] == dim)
          {
            // First let us do the last pseudo-row.
            // TODO
            // This is just anti-symmetry => optimize
            if (components[i] == dim && components[j] < dim)
            {
              copy_data.cell_matrix(i, j) += shape_value[i][q_point]
                * shape_grad[j][q_point][components[j]]
                * JxW[q_point];
            }
            else if (components[j] == dim && components[i] < dim)
            {
              copy_data.cell_matrix(i, j) -= shape_value[j][q_point]
                * shape_grad[i][q_point][components[i]]
                * JxW[q_point];
            }
          }
#pragma endregion

          // Magnetism forms - Laplace
          if (components[i] > dim && components[j] > dim)
          {
            // Laplace
            if (components[i] == components[j])
            {
              copy_data.cell_matrix(i, j) += shape_grad[i][q_point]
                * shape_grad[j][q_point]
                * JxW[q_point]
                / (MU * MU_R);
            }

#pragma region A_ONLY_LAPLACE
            if (!A_ONLY_LAPLACE && marker == MARKER_FLUID)
            {
              // (u x (\Nabla x A)) - first part (coinciding indices)
              if (components[i] == components[j])
              {
                for (int other_component = 0; other_component < dim; other_component++)
                {
                  if (other_component != components[i] - dim - 1)
                  {
                    copy_data.cell_matrix(i, j) += SIGMA[marker] * v_prev[q_point][other_component]
                      * shape_value[i][q_point]
                      * shape_grad[j][q_point][other_component]
                      * JxW[q_point];
                  }
                }
              }
              // (u x (\Nabla x A)) - second part (NON-coinciding indices)
              else
              {
                copy_data.cell_matrix(i, j) -= SIGMA[marker] * v_prev[q_point][components[j] - dim - 1]
                  * shape_value[i][q_point]
                  * shape_grad[j][q_point][components[i] - dim - 1]
                  * JxW[q_point];
              }
            }
#pragma endregion
          }
#pragma region A_ONLY_LAPLACE
          // But we must not forget to differentiate wrt. velocities
          if (!A_ONLY_LAPLACE && marker == MARKER_FLUID)
          {
            if (components[i] > dim && components[j] < dim && (components[i] != (components[j] + dim + 1)))
            {
              copy_data.cell_matrix(i, j) -= SIGMA[marker] * (old_solution_gradients[q_point][components[j] + dim + 1][components[i] - dim - 1] - old_solution_gradients[q_point][components[i]][components[j]])
                * shape_value[i][q_point]
                * shape_value[j][q_point]
                * JxW[q_point];
            }
          }
#pragma endregion
        }

        // Velocity rhs
        if (components[i] < dim)
        {
          copy_data.cell_rhs(i) += shape_grad[i][q_point]
            * old_solution_gradients[q_point][components[i]]
            * JxW[q_point]
            / REYNOLDS;

          // Pressure form
          copy_data.cell_rhs(i) -= shape_grad[i][q_point][components[i]]
            * old_solution_values[q_point][dim]
            * JxW[q_point];

          copy_data.cell_rhs(i) += shape_value[i][q_point]
            * old_solution_gradients[q_point][components[i]]
            * v_prev[q_point]
            * JxW[q_point];

#pragma region NO_MOVEMENT_INDUCED_FORCE
          if (!NO_MOVEMENT_INDUCED_FORCE && marker == MARKER_FLUID)
          {
            // Forces from magnetic field
            for (unsigned int j = 0; j < dim; ++j)
            {
              if (j == components[i])
              {
                for (int other_component = 0; other_component < dim; other_component++)
                {
                  if (other_component != components[i])
                  {
                    copy_data.cell_rhs(i) += SIGMA[marker] * C[q_point][other_component] * C[q_point][other_component]
                      * shape_value[i][q_point]
                      * v_prev[q_point][j]
                      * JxW[q_point];
                  }
                }
              }
              else
              {
                copy_data.cell_rhs(i) -= SIGMA[marker] * C[q_point][components[i]] * C[q_point][components[j]]
                  * shape_value[i][q_point]
                  * v_prev[q_point][j]
                  * JxW[q_point];
              }
            }
          }
#pragma endregion

#pragma region NO_EXT_CURR_DENSITY_FORCE
          if (!NO_EXT_CURR_DENSITY_FORCE && marker == MARKER_FLUID)
          {
            for (unsigned int j = 0; j < dim; ++j)
            {
              if (components[i] == j)
              {
                for (int other_component = 0; other_component < dim; other_component++)
                {
                  if (other_component != components[i])
                  {
                    double val = J_EXT(marker, other_component, q_p)
                      * shape_value[i][q_point]
                      * A_prev_gradients[q_point][j][other_component]
                      * JxW[q_point];

                    copy_data.cell_rhs(i) += val;
                  }
                }
              }
              // - second part (NON-coinciding indices)
              else
              {
                double val = J_EXT(marker, j, q_p)
                  * shape_value[i][q_point]
                  * A_prev_gradients[q_point][j][components[i]]
                  * JxW[q_point];
                copy_data.cell_rhs(i) -= val;
              }
            }
          }
#pragma endregion
        }
#pragma region PRESSURE
        // Pressure rhs
        if (components[i] == dim)
        {
          for (int vel_i = 0; vel_i < dim; vel_i++)
          {
            copy_data.cell_rhs(i) += shape_value[i][q_point]
              * old_solution_gradients[q_point][vel_i][vel_i]
              * JxW[q_point];
          }
        }
#pragma endregion

        // Magnetism rhs
        if (components[i] > dim)
        {
          // Laplace
          copy_data.cell_rhs(i) += shape_grad[i][q_point]
            * A_prev_gradients[q_point][components_mag_i]
            * JxW[q_point]
            / (MU * MU_R);

          // Remanent induction.
          if (marker == MARKER_MAGNET)
          {
            if (components_mag_i == 0) {
              copy_data.cell_rhs(i) += (B_R[2] * shape_grad[i][q_point][1] - B_R[1] * shape_grad[i][q_point][2])
                * JxW[q_point]
                / (MU * MU_R);
            }

            if (components_mag_i == 1) {
              copy_data.cell_rhs(i) += (B_R[0] * shape_grad[i][q_point][2] - B_R[2] * shape_grad[i][q_point][0])
                * JxW[q_point]
                / (MU * MU_R);
            }

            if (components_mag_i == 2) {
              copy_data.cell_rhs(i) += (B_R[1] * shape_grad[i][q_point][0] - B_R[0] * shape_grad[i][q_point][1])
                * JxW[q_point]
                / (MU * MU_R);
            }
          }

#pragma region A_ONLY_LAPLACE
          // Residual: u x (curl A)
          if (!A_ONLY_LAPLACE && marker == MARKER_FLUID)
          {
            for (unsigned int j = 0; j < dim; ++j)
            {
              if (components_mag_i != j)
              {
                copy_data.cell_rhs(i) -= SIGMA[marker] * (A_prev_gradients[q_point][j][components_mag_i] - A_prev_gradients[q_point][components_mag_i][j])
                  * shape_value[i][q_point]
                  * v_prev[q_point][j]
                  * JxW[q_point];
              }
            }
          }
#pragma endregion
        }
      }
    }

    // distribute local to global matrix
    copy_data.local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices(copy_data.local_dof_indices);

    copy_data.isAssembled = true;

    /*
    dealii::hp::FEFaceValues<dim> hp_fe_face_values(mappingCollection, feCollection, qCollectionFace, dealii::update_quadrature_points);

    for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
    {
    if (cell->face(face)->user_index() > 0)
    {
    hp_fe_face_values.reinit(cell, face);

    const dealii::FEFaceValues<dim> &fe_face_values = hp_fe_face_values.get_present_fe_values();
    const unsigned int n_face_q_points = fe_face_values.n_quadrature_points;

    std::vector<dealii::Vector<double> > old_solution_values_face(n_face_q_points, dealii::Vector<double>(COMPONENT_COUNT));
    std::vector<std::vector<Tensor<1, dim> > > old_solution_gradients_face(n_face_q_points, std::vector<dealii::Tensor<1, dim> >(COMPONENT_COUNT));

    fe_face_values.get_function_values(present_solution, old_solution_values_face);
    fe_face_values.get_function_gradients(present_solution, old_solution_gradients_face);

    for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
    {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
    {
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
    {
    }
    }
    }
    }
    }
    }
    */
  }

  template <int dim>
  void CustomSolver<dim>::copyLocalToGlobal(const AssemblyCopyData &copy_data)
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

  template <int dim>
  void CustomSolver<dim>::finishAssembling()
  {
    // Finally, we remove hanging nodes from the system and apply zero
    // boundary values to the linear system that defines the Newton updates
    // $\delta u^n$:
    hanging_node_constraints.condense(system_matrix);
    hanging_node_constraints.condense(system_rhs);

    ComponentMask velocity_mask(COMPONENT_COUNT, false);
    for (int i = 0; i < dim; i++)
      velocity_mask.set(i, true);

    ComponentMask magnetic_field_mask(COMPONENT_COUNT, false);
    for (int i = dim + 1; i < COMPONENT_COUNT; i++)
      magnetic_field_mask.set(i, true);

    std::map<types::global_dof_index, double> boundary_values;

    // Velocity
    for (std::vector<unsigned int>::iterator it = velocityDirichletMarkers.begin(); it != velocityDirichletMarkers.end(); ++it)
    {
      VectorTools::interpolate_boundary_values(dof_handler, *it, ZeroFunction<dim>(COMPONENT_COUNT), boundary_values, velocity_mask);
      MatrixTools::apply_boundary_values(boundary_values, system_matrix, newton_update, system_rhs);
    }
    if (INLET_VELOCITY_FIXED)
    {
      VectorTools::interpolate_boundary_values(dof_handler, INLET_VELOCITY_FIXED_BOUNDARY, ZeroFunction<dim>(COMPONENT_COUNT), boundary_values, velocity_mask);
      MatrixTools::apply_boundary_values(boundary_values, system_matrix, newton_update, system_rhs);
    }

    // Magnetism
    for (std::vector<unsigned int>::iterator it = magnetismDirichletMarkers.begin(); it != magnetismDirichletMarkers.end(); ++it)
    {
      VectorTools::interpolate_boundary_values(dof_handler, *it, ZeroFunction<dim>(COMPONENT_COUNT), boundary_values, magnetic_field_mask);
      MatrixTools::apply_boundary_values(boundary_values, system_matrix, newton_update, system_rhs);
    }
  }

  template <int dim>
  void CustomSolver<dim>::solveAlgebraicSystem(int inner_iteration)
  {
    if (PRINT_ALGEBRA)
    {
      std::cout << "  Printing system " << inner_iteration << "... " << std::endl;

      std::string matrix_file = "Matrix_";
      std::string rhs_file = "Rhs_";

      matrix_file.append(std::to_string(inner_iteration));
      rhs_file.append(std::to_string(inner_iteration));

      std::ofstream matrix_out(matrix_file);
      std::ofstream rhs_out(rhs_file);

      system_matrix.print(matrix_out);
      system_rhs.print(rhs_out, 3, true, false);

      matrix_out.close();
      rhs_out.close();
    }

    std::cout << "Solving..." << std::endl;

    direct_CustomSolver.initialize(system_matrix);

    // RHS for Newton is -F
    system_rhs *= -1.;
    direct_CustomSolver.vmult(newton_update, system_rhs);

    hanging_node_constraints.distribute(newton_update);

    present_solution.add(NEWTON_DAMPING, newton_update);
  }

  template<int dim>
  void CustomSolver<dim>::add_markers(typename Triangulation<dim>::cell_iterator cell)
  {
    // Volumetric.
    // 0 Left, 1 Right, 2 Bottom, 3 Top, 4 Front, 5 Back
    int layerFromEdge[6] = { 0, 0, 0, 0, 0, 0 };
    int comparedCoordinate[6] = { 0, 0, 1, 1, 2, 2 };
    double comparedValue[6] = { p1(0), p2(0), p1(1), p2(1), p1(2), p2(2) };

    for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
    {
      /*
      std::cout << "Face: " << face_number << ": [" <<
      cell->face(face_number)->center()(0) << ", " <<
      cell->face(face_number)->center()(1) << ", " <<
      cell->face(face_number)->center()(2) << "]" << std::endl;

      std::cout << std::fabs(cell->face(face_number)->center()(comparedCoordinate[face_number]) - comparedValue[face_number]) << std::endl;
      std::cout << singleLayerThickness(comparedCoordinate[face_number]) << std::endl;
      std::cout << std::fabs(cell->face(face_number)->center()(comparedCoordinate[face_number]) - comparedValue[face_number]) / singleLayerThickness(comparedCoordinate[face_number]) << std::endl;
      */
      layerFromEdge[face_number] = std::round(std::fabs(cell->face(face_number)->center()(comparedCoordinate[face_number]) - comparedValue[face_number]) / singleLayerThickness(comparedCoordinate[face_number]));
    }

    cell->set_material_id(MARKER_AIR);
    bool fluid = true;
    for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
    {
      if (i == 4 || i == 5)
        continue;
      if (layerFromEdge[i] < AIR_LAYER_THICKNESS + MAGNET_SIZE)
        fluid = false;
    }
    if (fluid)
      cell->set_material_id(MARKER_FLUID);

    bool magnet = true;
    for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
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
    for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
    {
      if (std::fabs(cell->face(face_number)->center()(2) - p1(2)) < 1e-12)
        cell->face(face_number)->set_boundary_indicator(BOUNDARY_FRONT);

      if (std::fabs(cell->face(face_number)->center()(0) - p1(0)) < 1e-12)
        cell->face(face_number)->set_boundary_indicator(BOUNDARY_LEFT);

      if (std::fabs(cell->face(face_number)->center()(2) - p2(2)) < 1e-12)
        cell->face(face_number)->set_boundary_indicator(BOUNDARY_BACK);

      if (std::fabs(cell->face(face_number)->center()(0) - p2(0)) < 1e-12)
        cell->face(face_number)->set_boundary_indicator(BOUNDARY_RIGHT);

      if (std::fabs(cell->face(face_number)->center()(1) - p1(1)) < 1e-12)
        cell->face(face_number)->set_boundary_indicator(BOUNDARY_BOTTOM);

      if (std::fabs(cell->face(face_number)->center()(1) - p2(1)) < 1e-12)
        cell->face(face_number)->set_boundary_indicator(BOUNDARY_TOP);

      if (((std::fabs(cell->face(face_number)->center()(0) - flowChannelBoundaries[0]) < 1e-12) || (std::fabs(cell->face(face_number)->center()(0) - flowChannelBoundaries[1]) < 1e-12)) && (std::fabs(cell->face(face_number)->center()(1)) > flowChannelBoundaries[0]) && (std::fabs(cell->face(face_number)->center()(1)) < flowChannelBoundaries[1]))
        cell->face(face_number)->set_boundary_indicator(BOUNDARY_VEL_WALL);

      if (((std::fabs(cell->face(face_number)->center()(1) - flowChannelBoundaries[0]) < 1e-12) || (std::fabs(cell->face(face_number)->center()(1) - flowChannelBoundaries[1]) < 1e-12)) && (std::fabs(cell->face(face_number)->center()(0)) > flowChannelBoundaries[0]) && (std::fabs(cell->face(face_number)->center()(0)) < flowChannelBoundaries[1]))
        cell->face(face_number)->set_boundary_indicator(BOUNDARY_VEL_WALL);
    }
  }

  template <int dim>
  void CustomSolver<dim>::init_discretization()
  {
    // Mesh
    GridGenerator::subdivided_hyper_rectangle(triangulation, refinements, p1, p2);

    typename Triangulation<dim>::cell_iterator
      cell = triangulation.begin(),
      endc = triangulation.end();
    for (; cell != endc; ++cell)
    {
      this->add_markers(cell);
    }

    // Set the coil cells to use the FE system where only magnetism is solved.
    typename dealii::hp::DoFHandler<dim>::active_cell_iterator dof_cell = dof_handler.begin_active(), dof_endc = dof_handler.end();
    for (; dof_cell != dof_endc; ++dof_cell)
    {
      if (dof_cell->material_id() != MARKER_FLUID)
      {
        dof_cell->set_active_fe_index(1);
      }
    }

    setup_system(true);
    set_boundary_values();
  }

  template <int dim>
  void CustomSolver<dim>::run()
  {
    init_discretization();

    if (PRINT_INIT_SLN)
    {
      std::cout << "  Printing initial solution... " << std::endl;

      DataOut<dim, hp::DoFHandler<dim> > data_out;

      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(present_solution, "solution");
      data_out.build_patches();
      const std::string filename = "Initial_Sln.vtk";
      std::ofstream output(filename.c_str());
      data_out.write_vtk(output);
    }

    // The Newton iteration starts next.
    double previous_res = 0.;

    for (unsigned int inner_iteration = 0; inner_iteration < NEWTON_ITERATIONS; ++inner_iteration)
    {
      std::cout << "Assembling..." << std::endl;
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

      output_results(inner_iteration);

      if (previous_res < NEWTON_RESIDUAL_THRESHOLD)
        break;
    }
  }
}

int main()
{
  try
  {
    using namespace dealii;
    using namespace Step15;

    deallog.depth_console(0);

    CustomSolver<DIM> fe_problem;
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
