#define DIM 3

const int INIT_REF_NUM = 3;
const double REYNOLDS = 5.;
const double NEWTON_DAMPING = 0.9;
const int NEWTON_ITERATIONS = 6;
const double INLET_AMPLITUDE = 1.0;

const int BOUNDARY_WALL = 0;
const int BOUNDARY_INLET = 1;
const int BOUNDARY_OUTLET = 2;

const bool INLET_FIXED = false;

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
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
namespace Step15
{
    using namespace dealii;

    template <int dim>
    class CustomSolver
    {
    public:
        CustomSolver(int polynomial_degree = 2);
        ~CustomSolver();

        void run();

    private:
        void setup_system(const bool initial_step);
        void assemble_system();
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

        int degree;
    };

    template <int dim>
    class BoundaryValuesInlet : public Function < dim >
    {
    public:
        // Function has to be called with the proper number of components
        BoundaryValuesInlet() : Function<dim>(dim + 1) {}

        virtual double value(const Point<dim>   &p, const unsigned int  component = 0) const;
    };

    template <>
    double BoundaryValuesInlet<2>::value(const Point<2> &p, const unsigned int component) const
    {
        if (component == 1)
        {
            return INLET_AMPLITUDE * (p(0) * (1.0 - p(0)));
        }
        else
            return 0;
    }

    template <>
    double BoundaryValuesInlet<3>::value(const Point<3> &p, const unsigned int component) const
    {
        if (component == 1)
        {
            //return INLET_AMPLITUDE * (p(0) * (1.0 - p(0))) * (p(2) * (1.0 - p(2)));
            return 0.;
        }
        else
            return 0;
    }

    template <int dim>
    class BoundaryValuesWall : public Function < dim >
    {
    public:
        // Function has to be called with the proper number of components
        BoundaryValuesWall() : Function<dim>(dim + 1) {}

        virtual double value(const Point<dim>   &p, const unsigned int  component = 0) const;
    };

    template <int dim>
    double BoundaryValuesWall<dim>::value(const Point<dim> &p, const unsigned int component) const
    {
        return 0;
    }

    template <int dim>
    CustomSolver<dim>::CustomSolver(int degree)
        : degree(degree), dof_handler(triangulation)
    {
        std::vector<const dealii::FiniteElement<dim> *> fes;
        std::vector<unsigned int> multiplicities;
        // Velocity
        fes.push_back(new dealii::FE_Q<dim>(degree));
        multiplicities.push_back(dim);

        // Pressure
        fes.push_back(new dealii::FE_Q<dim>(degree - 1));
        multiplicities.push_back(1);
        feCollection.push_back(dealii::FESystem<dim, dim>(fes, multiplicities));

        mappingCollection.push_back(dealii::MappingQ<dim>(1, true));

        // TODO
        // optimize, but the problem is the most consuming product is 2 * value, 1 * derivative which is basically this.
        qCollection.push_back(dealii::QGauss<dim>(3 * degree));
        qCollectionFace.push_back(dealii::QGauss<dim - 1>(3 * degree));
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
    }

    template <int dim>
    void CustomSolver<dim>::assemble_system()
    {
        system_matrix = 0;
        system_rhs = 0;

        dealii::hp::FEValues<dim> hp_fe_values(feCollection, qCollection, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

        dealii::hp::DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
        for (; cell != endc; ++cell)
        {
            hp_fe_values.reinit(cell);
            const dealii::FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
            const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
            const unsigned int n_q_points = fe_values.n_quadrature_points;

            FullMatrix<double>           cell_matrix(dofs_per_cell, dofs_per_cell);
            Vector<double>               cell_rhs(dofs_per_cell);

            std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);

            std::vector<dealii::Vector<double> > old_solution_values(n_q_points, dealii::Vector<double>(dim + 1));
            std::vector<dealii::Tensor<1, dim> > old_velocity_values(n_q_points, dealii::Tensor<1, dim>(dim));
            std::vector<std::vector<Tensor<1, dim> > > old_solution_gradients(n_q_points, std::vector<dealii::Tensor<1, dim> >(dim + 1));

            fe_values.get_function_values(present_solution, old_solution_values);
            // Only velocity values for simpler form expressions
            for (int i = 0; i < old_solution_values.size(); i++)
            {
                for (int j = 0; j < dim; j++)
                    old_velocity_values[i][j] = old_solution_values[i][j];
            }
            fe_values.get_function_gradients(present_solution, old_solution_gradients);

            std::vector<int> components(dofs_per_cell);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                components[i] = cell->get_fe().system_to_component_index(i).first;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        //std::cout << fe_values.shape_grad(i, q_point) << std::endl;
                        //std::cout << fe_values.shape_grad(j, q_point) << std::endl;
                        //double res = fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point);
                        //std::cout << res << std::endl;

                        // Velocity forms
                        if (components[i] < dim && components[j] < dim)
                        {
                            if (components[i] == components[j])
                            {
                                // Diffusion
                                cell_matrix(i, j) += fe_values.shape_grad(i, q_point)
                                    * fe_values.shape_grad(j, q_point)
                                    * fe_values.JxW(q_point)
                                    / REYNOLDS;

                                // Advection - 1/2
                                // result += wt[i] * ((xvel_prev_newton->val[i] * u->dx[i] + yvel_prev_newton->val[i]
                                // *u->dy[i]) * v->val[i] ....
                                cell_matrix(i, j) += old_velocity_values[q_point]
                                    * fe_values.shape_grad(j, q_point)
                                    * fe_values.shape_value(i, q_point)
                                    * fe_values.JxW(q_point);

                                // Advection - 2/2
                                // ... + u->val[i] * v->val[i] * xvel_prev_newton->dx[i]);
                                cell_matrix(i, j) += old_solution_gradients[q_point][components[i]][components[i]]
                                    * fe_values.shape_value(i, q_point)
                                    * fe_values.shape_value(j, q_point)
                                    * fe_values.JxW(q_point);
                            }
                            // Nonsymmetrical terms
                            else
                            {
                                cell_matrix(i, j) += old_solution_gradients[q_point][components[i]][components[j]]
                                    * fe_values.shape_value(i, q_point)
                                    * fe_values.shape_value(j, q_point)
                                    * fe_values.JxW(q_point);
                            }
                        }
                        // Pressure forms
                        else
                        {
                            // First let us do the last pseudo-row.
                            // TODO
                            // This is just anti-symmetry => optimize
                            if (components[i] == dim && components[j] < dim)
                            {
                                cell_matrix(i, j) += fe_values.shape_value(i, q_point)
                                    * fe_values.shape_grad(j, q_point)[components[j]]
                                    * fe_values.JxW(q_point);
                            }
                            else if (components[j] == dim && components[i] < dim)
                            {
                                cell_matrix(i, j) -= fe_values.shape_value(j, q_point)
                                    * fe_values.shape_grad(i, q_point)[components[i]]
                                    * fe_values.JxW(q_point);
                            }
                        }
                    }

                    if (components[i] < dim)
                    {
                        // result += wt[i] * ((xvel_prev_newton->dx[i] * v->dx[i] + xvel_prev_newton->dy[i] * v->dy[i]) / Reynolds - (p_prev_newton->val[i] * v->dx[i]));
                        cell_rhs(i) -= fe_values.shape_grad(i, q_point)
                            * old_solution_gradients[q_point][components[i]]
                            * fe_values.JxW(q_point)
                            / REYNOLDS;

                        cell_rhs(i) += fe_values.shape_grad(i, q_point)[components[i]]
                            * old_solution_values[q_point][dim]
                            * fe_values.JxW(q_point);

                        // result += wt[i] * (xvel_prev_newton->val[i] * xvel_prev_newton->dx[i] + yvel_prev_newton->val[i] * xvel_prev_newton->dy[i]) * v->val[i];
                        cell_rhs(i) -= fe_values.shape_value(i, q_point)
                            * old_solution_gradients[q_point][components[i]]
                            * old_velocity_values[q_point]
                            * fe_values.JxW(q_point);

                        // Force - upward
                        if (components[i] == 1)
                            cell_rhs(i) += fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
                    }
                    else
                    {
                        // result += wt[i] * (xvel_prev_newton->dx[i] * v->val[i] + yvel_prev_newton->dy[i] * v->val[i]);
                        for (int vel_i = 0; vel_i < dim; vel_i++)
                        {
                            cell_rhs(i) -= fe_values.shape_value(i, q_point)
                                * old_solution_gradients[q_point][vel_i][vel_i]
                                * fe_values.JxW(q_point);
                        }
                    }
                }
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    system_matrix.add(local_dof_indices[i],
                        local_dof_indices[j],
                        cell_matrix(i, j));
                }
                system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }

            dealii::hp::FEFaceValues<dim> hp_fe_face_values(mappingCollection, feCollection, qCollectionFace, dealii::update_quadrature_points);

            for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
            {
                if (cell->face(face)->user_index() > 0)
                {
                    hp_fe_face_values.reinit(cell, face);

                    const dealii::FEFaceValues<dim> &fe_face_values = hp_fe_face_values.get_present_fe_values();
                    const unsigned int n_face_q_points = fe_face_values.n_quadrature_points;

                    std::vector<dealii::Vector<double> > old_solution_values_face(n_face_q_points, dealii::Vector<double>(dim + 1));
                    std::vector<dealii::Tensor<1, dim> > old_velocity_values_face(n_face_q_points, dealii::Tensor<1, dim>(dim));
                    std::vector<std::vector<Tensor<1, dim> > > old_solution_gradients_face(n_face_q_points, std::vector<dealii::Tensor<1, dim> >(dim + 1));

                    fe_face_values.get_function_values(present_solution, old_solution_values_face);
                    // Only velocity values for simpler form expressions
                    for (int i = 0; i < old_solution_values_face.size(); i++)
                    {
                        for (int j = 0; j < dim; j++)
                            old_velocity_values_face[i][j] = old_solution_values_face[i][j];
                    }
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

        }

        // Finally, we remove hanging nodes from the system and apply zero
        // boundary values to the linear system that defines the Newton updates
        // $\delta u^n$:
        hanging_node_constraints.condense(system_matrix);
        hanging_node_constraints.condense(system_rhs);

        FEValuesExtractors::Vector velocities(0);
        ComponentMask velocities_mask = this->feCollection.component_mask(velocities);

        std::map<types::global_dof_index, double> boundary_values_wall;
        std::map<types::global_dof_index, double> boundary_values_inlet;

        VectorTools::interpolate_boundary_values(dof_handler, BOUNDARY_WALL, ZeroFunction<dim>(dim + 1), boundary_values_wall, velocities_mask);
        MatrixTools::apply_boundary_values(boundary_values_wall, system_matrix, newton_update, system_rhs);

        if (INLET_FIXED)
        {
            VectorTools::interpolate_boundary_values(dof_handler, BOUNDARY_INLET, ZeroFunction<dim>(dim + 1), boundary_values_inlet, velocities_mask);
            MatrixTools::apply_boundary_values(boundary_values_inlet, system_matrix, newton_update, system_rhs);
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
        direct_CustomSolver.vmult(newton_update, system_rhs);

        hanging_node_constraints.distribute(newton_update);

        present_solution.add(NEWTON_DAMPING, newton_update);
    }

    template <int dim>
    void CustomSolver<dim>::set_boundary_values()
    {
        std::map<types::global_dof_index, double> boundary_values_wall;
        std::map<types::global_dof_index, double> boundary_values_inlet;

        FEValuesExtractors::Vector velocities(0);
        ComponentMask velocities_mask = this->feCollection.component_mask(velocities);

        VectorTools::interpolate_boundary_values(dof_handler, BOUNDARY_WALL, BoundaryValuesWall<dim>(), boundary_values_wall, velocities_mask);
        for (std::map<types::global_dof_index, double>::const_iterator p = boundary_values_wall.begin(); p != boundary_values_wall.end(); ++p)
            present_solution(p->first) = p->second;

        if (INLET_FIXED)
        {
            VectorTools::interpolate_boundary_values(dof_handler, BOUNDARY_INLET, BoundaryValuesInlet<dim>(), boundary_values_inlet, velocities_mask);
            for (std::map<types::global_dof_index, double>::const_iterator p = boundary_values_inlet.begin(); p != boundary_values_inlet.end(); ++p)
                present_solution(p->first) = p->second;
        }
    }

    template <int dim>
    void CustomSolver<dim>::run()
    {
        // Mesh
        GridGenerator::hyper_cube(triangulation);
        triangulation.refine_global(INIT_REF_NUM);

        typename Triangulation<dim>::cell_iterator
            cell = triangulation.begin(),
            endc = triangulation.end();
        for (; cell != endc; ++cell)
        {
            for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
            {
                if (std::fabs(cell->face(face_number)->center()(1)) < 1e-12)
                    cell->face(face_number)->set_boundary_indicator(BOUNDARY_INLET);

                if (std::fabs(cell->face(face_number)->center()(1) - 1.0) < 1e-12)
                    cell->face(face_number)->set_boundary_indicator(BOUNDARY_OUTLET);
            }
        }

        // The Newton iteration starts next.
        double previous_res = 0;
        setup_system(true);
        set_boundary_values();

        for (unsigned int inner_iteration = 0; inner_iteration < NEWTON_ITERATIONS; ++inner_iteration)
        {
            assemble_system();
            // system_rhs.print(std::cout);
            // system_matrix.print(std::cout);
            previous_res = system_rhs.l2_norm();
            solve();
            std::cout << "  Residual: "
                << previous_res
                << std::endl;
        }

        // Every fifth iteration, i.e., just before we refine the mesh again,
        // we output the solution as well as the Newton update. This happens
        // as in all programs before:
        DataOut<dim, hp::DoFHandler<dim> > data_out;

        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(present_solution, "solution");
        data_out.add_data_vector(newton_update, "update");
        data_out.build_patches();
        const std::string filename = "solution.vtk";
        std::ofstream output(filename.c_str());
        data_out.write_vtk(output);
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
