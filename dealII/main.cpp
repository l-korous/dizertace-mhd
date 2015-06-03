// A comes after v, p in the system (hardcoded)
// Assembling is done in the form J() | = F()
// i.e. the negative sign on the RHS is handled after assembling of F.

#define DIM 3

#pragma region TESTING

const bool PRINT_ALGEBRA = false;
const bool A_ONLY_LAPLACE = false;
const bool NO_MOVEMENT_INDUCED_FORCE = true;
const bool NO_EXT_CURR_DENSITY_FORCE = false;
const bool A_LINEAR_WRT_Z = true;

#pragma endregion

#include <deal.II/base/point.h>
#include <vector>
const dealii::Point<DIM> p1(0., 0., 0.);
const dealii::Point<DIM> p2(1., 1., 1.);
const unsigned int INIT_REF_NUM = 4;
const std::vector<unsigned int> refinements({ INIT_REF_NUM, INIT_REF_NUM, INIT_REF_NUM });

const unsigned int BOUNDARY_FRONT = 1;
const unsigned int BOUNDARY_RIGHT = 2;
const unsigned int BOUNDARY_BACK = 3;
const unsigned int BOUNDARY_LEFT = 4;
const unsigned int BOUNDARY_BOTTOM = 5;
const unsigned int BOUNDARY_TOP = 6;

std::vector<unsigned int> velocityDirichletMarkers({ BOUNDARY_FRONT, BOUNDARY_RIGHT, BOUNDARY_BACK, BOUNDARY_LEFT });
std::vector<unsigned int> magnetismDirichletMarkers({ BOUNDARY_FRONT, BOUNDARY_BOTTOM, BOUNDARY_BACK, BOUNDARY_TOP });

const bool INLET_VELOCITY_FIXED = false;
const double INLET_VELOCITY_AMPLITUDE = 1.0;

const unsigned int POLYNOMIAL_DEGREE_MAG = 2;

const double MU = 1.2566371e-6;
const double MU_R = 1.;
const double SIGMA = 3.e6;
const double A_0[3] = { 1.e-1, 0., 0. };
const double J_EXT[3] = { 1.e5, 0., 0. };

const double REYNOLDS = 5.;

const double NEWTON_DAMPING = 1.;
const int NEWTON_ITERATIONS = 30;

const int COMPONENT_COUNT = 2 * DIM + 1;

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
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
  class CustomSolver
  {
  public:
    CustomSolver();
    ~CustomSolver();

    void run();

  private:
    void setup_system(const bool initial_step);
    void assemble_system();

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

    void localAssembleSystem(const typename dealii::hp::DoFHandler<dim>::active_cell_iterator &iter,
      AssemblyScratchData &scratch_data,
      AssemblyCopyData &copy_data);

    void copyLocalToGlobal(const AssemblyCopyData &copy_data);

    void finishAssembling();

    void solve();
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

  template <int dim>
  class BoundaryValuesInlet : public Function < dim >
  {
  public:
    // Function has to be called with the proper number of components
    BoundaryValuesInlet() : Function<dim>(COMPONENT_COUNT) {}

    virtual double value(const Point<dim>   &p, const unsigned int  component = 0) const;
  };

  template <>
  double BoundaryValuesInlet<2>::value(const Point<2> &p, const unsigned int component) const
  {
    if (component == 1)
      return INLET_VELOCITY_AMPLITUDE * (p(0) * (1.0 - p(0)));
    else
      return 0;
  }

  template <>
  double BoundaryValuesInlet<3>::value(const Point<3> &p, const unsigned int component) const
  {
    if (component == 1)
      return INLET_VELOCITY_AMPLITUDE * (p(0) * (1.0 - p(0))) * (p(2) * (1.0 - p(2)));
    else
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
    if (component > dim)
    {
      double coefficient = A_LINEAR_WRT_Z ? (p(2) - p1(2)) / (p2(2) - p1(2)) : 1.;
      return A_0[component - dim - 1] * coefficient;
    }
    else
      return 0;
  }

  template <int dim>
  CustomSolver<dim>::CustomSolver() : dof_handler(triangulation)
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

    // Push all components
    feCollection.push_back(dealii::FESystem<dim, dim>(fes, multiplicities));

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

  // @sect4{CustomSolver::setup_system}

  // As always in the setup-system function, we setup the variables of the
  // finite element method. There are same differences to step-6, because
  // there we start solving the PDE from scratch in every refinement cycle
  // whereas here we need to take the solution from the previous mesh onto the
  // current mesh. Consequently, we can't just reset solution vectors. The
  // argument passed to this function thus indicates whether we can
  // distributed degrees of freedom (plus compute constraints) and set the
  // solution vector to zero or whether this has happened elsewhere already
  // (specifically, in <code>refine_mesh()</code>).
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
      AssemblyScratchData(this->feCollection, this->mappingCollection, this->qCollection),
      AssemblyCopyData());

    this->finishAssembling();
  }


  template <int dim>
  CustomSolver<dim>::AssemblyScratchData::AssemblyScratchData(const dealii::hp::FECollection<dim> &feCollection,
    const dealii::hp::MappingCollection<dim> &mappingCollection,
    const dealii::hp::QCollection<dim> &quadratureFormulas)
    :
    hp_fe_values(mappingCollection, feCollection, quadratureFormulas,
    dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values)
  {}

  template <int dim>
  CustomSolver<dim>::AssemblyScratchData::AssemblyScratchData(const AssemblyScratchData &scratch_data)
    :
    hp_fe_values(scratch_data.hp_fe_values.get_mapping_collection(),
    scratch_data.hp_fe_values.get_fe_collection(),
    scratch_data.hp_fe_values.get_quadrature_collection(),
    dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values)
  {}

  template <int dim>
  CustomSolver<dim>::AssemblyCopyData::AssemblyCopyData()
    : isAssembled(false), cell_matrix(0), cell_rhs(0)
  {}

  template <int dim>
  void CustomSolver<dim>::localAssembleSystem(const typename dealii::hp::DoFHandler<dim>::active_cell_iterator &cell,
    AssemblyScratchData &scratch_data,
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

    std::vector<int> components(dofs_per_cell);

    fe_values.get_function_values(present_solution, old_solution_values);
    fe_values.get_function_gradients(present_solution, old_solution_gradients);

    // Only velocity values for simpler form expressions
    for (int i = 0; i < old_solution_values.size(); i++)
    {
      for (int j = 0; j < dim; j++)
        v_prev[i][j] = old_solution_values[i][j];
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

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
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

              if (!NO_MOVEMENT_INDUCED_FORCE)
              {
                // sigma (u x B) x B WRT VELOCITIES - coinciding indices
                for (int other_component = 0; other_component < dim; other_component++)
                {
                  if (other_component != components[i])
                  {
                    copy_data.cell_matrix(i, j) += SIGMA * C[q_point][other_component] * C[q_point][other_component]
                      * shape_value[i][q_point]
                      * shape_value[j][q_point]
                      * JxW[q_point];
                  }
                }
              }
            }
            // NON-Coinciding indices.
            else
            {
              // Nonsymmetrical terms from N-S equations
              copy_data.cell_matrix(i, j) += old_solution_gradients[q_point][components[i]][components[j]]
                * shape_value[i][q_point]
                * shape_value[j][q_point]
                * JxW[q_point];

              if (!NO_MOVEMENT_INDUCED_FORCE)
              {
                // sigma (u x B) x B WRT VELOCITIES - NON-coinciding indices
                copy_data.cell_matrix(i, j) -= SIGMA * C[q_point][components[i]] * C[q_point][components[j]]
                  * shape_value[i][q_point]
                  * shape_value[j][q_point]
                  * JxW[q_point];
              }
            }
          }

          // [J_{ext} + Sigma (u x B)] x B WRT MAGNETISM
          if (components[i] < dim && components[j] > dim)
          {
            if (!NO_MOVEMENT_INDUCED_FORCE)
            {
#pragma region paperToCodeHelpers
  #define v_x v_prev[q_point][0]
  #define v_y v_prev[q_point][1]
  #define v_z v_prev[q_point][2]

  #define P_x_y old_solution_gradients[q_point][4][1]
  #define P_x_z old_solution_gradients[q_point][4][2]

  #define P_y_x old_solution_gradients[q_point][5][0]
  #define P_y_z old_solution_gradients[q_point][5][2]

  #define P_z_x old_solution_gradients[q_point][6][0]
  #define P_z_y old_solution_gradients[q_point][6][1]

  #define C_x C[q_point][0]
  #define C_y C[q_point][1]
  #define C_z C[q_point][2]

  #define u_x shape_grad[j][q_point][0]
  #define u_y shape_grad[j][q_point][1]
  #define u_z shape_grad[j][q_point][2]
#pragma endregion
              double value = 0.;
              switch (components[i])
              {
              case 0:
                switch (components[j] - dim - 1)
                {
                case 0:
                  value = -(v_z * C_x * u_y) - (v_x * (2. * P_x_y * u_y - 2. * P_y_x * u_y)) - (v_x * (2. * P_x_z * u_z - 2. * P_z_x * u_z)) + (v_y * C_x * u_z);
                  break;
                case 1:
                  value = (v_z * (P_z_y * u_x - (P_y_x * u_z + P_y_z * u_x) + P_x_y * u_z)) - (v_x * (2. * P_y_x * u_x - 2. * P_x_y * u_x)) - (v_y * C_y * u_z);
                  break;
                case 2:
                  value = (v_z * C_z * u_y) - (v_x * (2. * P_z_x * u_x - 2. * P_x_z * u_x)) + (v_y * (P_x_z * u_y - (P_z_x * u_y + P_z_y * u_x) + P_y_z * u_x));
                  break;
                }
                break;
                break;
              case 1:
                switch (components[j] - dim - 1)
                {
                case 0:
                  value = (v_x * C_x * u_z) - (v_y * (2. * P_x_y * u_y - 2. * P_y_x * u_y)) + (v_z * (P_y_x * u_z - (P_x_z * u_y + P_x_y * u_z) + P_z_x * u_y));
                  break;
                case 1:
                  value = (-v_x * C_y * u_z) - (v_y * (2. * P_y_z * u_z - 2. * P_z_y * u_z)) - (v_y * (2.* P_y_x * u_x - 2. * P_x_y * u_x)) + (v_z * C_y * u_x);
                  break;
                case 2:
                  value = (v_x * (P_x_z * u_y - (P_z_y * u_x + P_z_x * u_y) + P_y_z * u_x)) - (v_y * (2. * P_z_y * u_y - 2. * P_y_z * u_y)) - (v_z * C_z * u_x);
                    break;
                }
                break;
                break;
              case 2:
                switch (components[j] - dim - 1)
                {
                case 0:
                    value = (v_y * (P_y_x * u_y - (P_x_y * u_z + P_x_z * u_y) + P_z_x * u_y)) - (v_z * (2. * P_x_z * u_z - 2. * P_z_x * u_z)) - (v_x * C_x * u_y);
                  break;
                case 1:
                  value = (v_y * C_y * u_x) - (v_z * (2. * P_y_z * u_z - 2. * P_z_y * u_z)) + (v_x * (P_z_y * u_x - (P_y_z * u_x + P_y_x * u_z) + P_x_y * u_z));
                  break;
                case 2:
                  value = (-v_y * C_z * P_z_x) - (v_z * (2. * P_z_x * u_x - 2. * P_x_z * u_x)) - (v_z * (2. * P_z_y * u_y - 2. * P_y_z * u_y)) + (v_x * C_z * u_y);
                  break;
                }
                break;
              }
              copy_data.cell_matrix(i, j) -= SIGMA * value
                * shape_value[i][q_point]
                * JxW[q_point];
            }

            if (!NO_EXT_CURR_DENSITY_FORCE)
            {
              // (J_ext x (\Nabla x A))
              // - first part (coinciding indices)
              if (components[i] == components[j] - dim - 1)
              {
                for (int other_component = 0; other_component < dim; other_component++)
                {
                  if (other_component != components[i])
                  {
                    copy_data.cell_matrix(i, j) += SIGMA * J_EXT[other_component]
                      * shape_value[i][q_point]
                      * shape_grad[j][q_point][other_component]
                      * JxW[q_point];
                  }
                }
              }
              // - second part (NON-coinciding indices)
              else
              {
                copy_data.cell_matrix(i, j) -= SIGMA * J_EXT[components[j] - dim - 1]
                  * shape_value[i][q_point]
                  * shape_grad[j][q_point][components[i]]
                  * JxW[q_point];
              }
            }
          }

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

          // Magnetism forms - Laplace
          if (components[i] > dim && components[j] > dim)
          {
            // Laplace
            if (components[i] == components[j])
            {
              copy_data.cell_matrix(i, j) -= shape_grad[i][q_point]
                * shape_grad[j][q_point]
                * JxW[q_point]
                / (MU * MU_R);
            }

            if (!A_ONLY_LAPLACE)
            {
              // (u x (\Nabla x A)) - first part (coinciding indices)
              if (components[i] == components[j])
              {
                for (int other_component = 0; other_component < dim; other_component++)
                {
                  if (other_component != components[i] - dim - 1)
                  {
                    copy_data.cell_matrix(i, j) += SIGMA * v_prev[q_point][other_component]
                      * shape_value[i][q_point]
                      * shape_grad[j][q_point][other_component]
                      * JxW[q_point];
                  }
                }
              }
              // (u x (\Nabla x A)) - second part (NON-coinciding indices)
              else
              {
                copy_data.cell_matrix(i, j) -= SIGMA * v_prev[q_point][components[j] - dim - 1]
                  * shape_value[i][q_point]
                  * shape_grad[j][q_point][components[i] - dim - 1]
                  * JxW[q_point];
              }
            }
          }
          // But we must not forget to differentiate wrt. velocities
          if (!A_ONLY_LAPLACE)
          {
            if (components[i] > dim && components[j] < dim && (components[i] != (components[j] + dim + 1)))
            {
              copy_data.cell_matrix(i, j) -= SIGMA * (old_solution_gradients[q_point][components[j] + dim + 1][components[i] - dim - 1] - old_solution_gradients[q_point][components[i]][components[j]])
                * shape_value[i][q_point]
                * shape_value[j][q_point]
                * JxW[q_point];
            }
          }
        }

        // Velocity rhs
        if (components[i] < dim)
        {
          copy_data.cell_rhs(i) += shape_grad[i][q_point]
            * old_solution_gradients[q_point][components[i]]
            * JxW[q_point]
            / REYNOLDS;

          copy_data.cell_rhs(i) -= shape_grad[i][q_point][components[i]]
            * old_solution_values[q_point][dim]
            * JxW[q_point];

          copy_data.cell_rhs(i) += shape_value[i][q_point]
            * old_solution_gradients[q_point][components[i]]
            * v_prev[q_point]
            * JxW[q_point];

          if (!NO_MOVEMENT_INDUCED_FORCE)
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
                    copy_data.cell_rhs(i) += SIGMA * C[q_point][other_component] * C[q_point][other_component]
                      * shape_value[i][q_point]
                      * v_prev[q_point][j]
                      * JxW[q_point];
                  }
                }
              }
              else
              {
                copy_data.cell_rhs(i) -= SIGMA * C[q_point][components[i]] * C[q_point][components[j]]
                  * shape_value[i][q_point]
                  * v_prev[q_point][j]
                  * JxW[q_point];
              }
            }
          }

          // External force.
          if (!NO_EXT_CURR_DENSITY_FORCE)
          {
            for (unsigned int j = 0; j < dim; ++j)
            {
              if (components[i] < dim && (components[i] != j))
              {
                copy_data.cell_rhs(i) -= (old_solution_gradients[q_point][j + dim + 1][components[i]] - old_solution_gradients[q_point][components[i] + dim + 1][j])
                  * shape_value[i][q_point]
                  * J_EXT[j]
                  * JxW[q_point];
              }
            }
          }
        }
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

        // Magnetism rhs
        if (components[i] > dim)
        {
          // Laplace
          copy_data.cell_rhs(i) += shape_grad[i][q_point]
            * old_solution_gradients[q_point][components[i]]
            * JxW[q_point]
            / (MU * MU_R);

          if (!A_ONLY_LAPLACE)
          {
            // External current density.
            // This is with a minus sign, because in F(u) = 0 form, it is on the left hand side.
            copy_data.cell_rhs(i) -= shape_value[i][q_point]
              * J_EXT[components[i] - dim - 1]
              * JxW[q_point];
          }

          // Residual: u x (curl A)
          if (!A_ONLY_LAPLACE)
          {
            for (unsigned int j = 0; j < dim; ++j)
            {
              if (components[i] > dim && (components[i] != (j + dim + 1)))
              {
                copy_data.cell_rhs(i) -= SIGMA * (old_solution_gradients[q_point][j + dim + 1][components[i] - dim - 1] - old_solution_gradients[q_point][components[i]][j])
                  * shape_value[i][q_point]
                  * v_prev[q_point][j]
                  * JxW[q_point];
              }
            }
          }
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
      VectorTools::interpolate_boundary_values(dof_handler, BOUNDARY_BOTTOM, ZeroFunction<dim>(COMPONENT_COUNT), boundary_values, velocity_mask);
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
  void CustomSolver<dim>::set_boundary_values()
  {
    std::map<types::global_dof_index, double> boundary_values;

    ComponentMask velocity_mask(COMPONENT_COUNT, false);
    for (int i = 0; i < dim; i++)
      velocity_mask.set(i, true);

    ComponentMask magnetic_field_mask(COMPONENT_COUNT, false);
    for (int i = dim + 1; i < COMPONENT_COUNT; i++)
      magnetic_field_mask.set(i, true);

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



  // @sect4{CustomSolver::solve}

  // The solve function is the same as always. At the end of the solution
  // process we update the current solution by setting
  // $u^{n+1}=u^n+\alpha^n\;\delta u^n$.
  template <int dim>
  void CustomSolver<dim>::solve()
  {
    direct_CustomSolver.initialize(system_matrix);

    // RHS for Newton is -F
    system_rhs *= -1;
    direct_CustomSolver.vmult(newton_update, system_rhs);

    hanging_node_constraints.distribute(newton_update);

    present_solution.add(NEWTON_DAMPING, newton_update);
  }

  template <int dim>
  void CustomSolver<dim>::run()
  {
    // Mesh
    GridGenerator::subdivided_hyper_rectangle(triangulation, refinements, p1, p2);

    typename Triangulation<dim>::cell_iterator
      cell = triangulation.begin(),
      endc = triangulation.end();
    for (; cell != endc; ++cell)
    {
      for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
      {
        if (std::fabs(cell->face(face_number)->center()(1) - p1(1)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator(BOUNDARY_FRONT);

        if (std::fabs(cell->face(face_number)->center()(0) - p1(0)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator(BOUNDARY_LEFT);

        if (std::fabs(cell->face(face_number)->center()(1) - p2(1)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator(BOUNDARY_BACK);

        if (std::fabs(cell->face(face_number)->center()(0) - p2(0)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator(BOUNDARY_RIGHT);

        if (std::fabs(cell->face(face_number)->center()(2) - p1(2)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator(BOUNDARY_BOTTOM);

        if (std::fabs(cell->face(face_number)->center()(2) - p2(2)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator(BOUNDARY_TOP);
      }
    }

    // The Newton iteration starts next.
    double previous_res = 0;
    setup_system(true);
    set_boundary_values();

    if (PRINT_ALGEBRA)
    {
      std::cout << "  Printing initial solution... " << std::endl;

      std::ofstream sln_out("Initial_Sln");
      present_solution.print(sln_out, 3, true, false);
      sln_out.close();
    }


    for (unsigned int inner_iteration = 0; inner_iteration < NEWTON_ITERATIONS; ++inner_iteration)
    {
      std::cout << "Assembling..." << std::endl;
      assemble_system();
      previous_res = system_rhs.l2_norm();
      std::cout << "  Residual: " << previous_res << std::endl;

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

      solve();

      if (PRINT_ALGEBRA)
      {
        std::cout << "  Printing solution " << inner_iteration << "... " << std::endl;

        std::string sln_file = "Sln_";
        sln_file.append(std::to_string(inner_iteration));
        std::ofstream sln_out(sln_file);
        newton_update.print(sln_out, 8, true, false);
        sln_out.close();
      }

      DataOut<dim, hp::DoFHandler<dim> > data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(present_solution, "solution");
      data_out.build_patches();
      std::string filename = "solution";
      filename.append(std::to_string(inner_iteration));
      filename.append(".vtk");
      std::ofstream output(filename.c_str());
      data_out.write_vtk(output);
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
