#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/lac/precondition.h>
//#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <map>

#include "Parameter.h"
#include "VEM_explicit.h"

#define Viscoelasticity_qp_data std::vector<std::vector<Viscoelasticity::Data<dim>>>

//-----------------------------------------------------------------------------------

using namespace dealii;

template <int dim>
class Solid
{

public:

	Solid(std::string filename_parameters);

	virtual ~Solid(	);

	void run();


private:
	
	void make_grid();

	void system_setup();

	void assemble_system(Viscoelasticity_qp_data &qp_data,
						 Viscoelasticity::VEM<dim> &material);

	void reset_viscoelasticity(Viscoelasticity_qp_data &qp_data);

	void make_constraints(const int &it_nr);

	void solve_load_step_NR(Viscoelasticity_qp_data &qp_data,
	                        Viscoelasticity::VEM<dim> &material, Vector<double> &solution_delta);

	std::pair<unsigned int,double> solve_linear_system(Vector<double> &newton_update);

	Vector<double> get_total_solution(const Vector<double> &solution_delta) const;

	void output_results(Viscoelasticity_qp_data &viscoplasticity_gauss_point_data,
	                    Viscoelasticity::VEM<dim> &material);

	void compute_stress_projection_and_average(Vector<double> &displacement,
	                                           Vector<double> &stress_projected,
	                                           Vector<double> &stress_averaged,
	                                           Viscoelasticity_qp_data &viscoplasticity_gauss_point_data,
	                                           Viscoelasticity::VEM<dim> &material);
	
	Triangulation<dim>                 triangulation;

	const FESystem<dim>                fe;

	DoFHandler<dim>                    dof_handler;

	const unsigned int                 dofs_per_cell;

	const FEValuesExtractors::Vector   u_fe;
	
	const QGauss<dim>                  qf_cell;

	const unsigned int                 n_q_points;

//	AffineConstraints<double>          constraints;
	ConstraintMatrix 				constraints;

	SparsityPattern					   sparsity_pattern;

	SparseMatrix<double>               tangent_matrix;

	Vector<double>              	   system_rhs;

	Vector<double>                     solution_n;

	Vector<double>				       solution_delta;

	unsigned int 					   current_load_step=0;

	unsigned int 					   max_number_newton_iterations=10;
	
	const Parameter::GeneralParameters parameter;

	std::vector<std::pair<double,std::string>> times_and_names;

	struct Errors
	{
		Errors() : u(1.0) {}

		void reset() {u = 1.0;}

		void normalise(const Errors &err) {if (err.u != 0.0) u /= err.u;}

		double u;
	};

	Errors error_residual, error_residual_0, error_residual_norm;
			
	void get_error_residual(Errors &error_residual);

	void print_conv_header();

	void print_conv_footer();

	double time;
	double total_displacement = 0.0;
	//-------------------------------------------------------------------------		
};



// Ludwig
template <int dim>
Solid<dim>::Solid(std::string filename_parameters)
  :
  fe(FE_Q<dim>(1),dim),
  dof_handler(triangulation),
  dofs_per_cell(fe.dofs_per_cell),
  u_fe(0),
  qf_cell(fe.degree+1),
  n_q_points(qf_cell.size()),
  parameter(filename_parameters),
  time(0)
{
}



template <int dim>
Solid<dim>::~Solid()
{
	dof_handler.clear();
}

template <int dim>
void Solid<dim>::reset_viscoelasticity(Viscoelasticity_qp_data &qp_data)
{
  for (unsigned int i = 0; i < triangulation.n_cells(); ++i)
	  for (unsigned int j = 0; j < n_q_points; ++j)
	  {
		  for (unsigned int k = 0; k < dim; ++k)
		  {
			  qp_data[i][j].Fv_A_0[k][k] = 1.0;
			  qp_data[i][j].Fv_A_last_timestep[k][k] = 1.0;
			  qp_data[i][j].F0[k][k] = 1.0;
			  qp_data[i][j].F0_last_timestep[k][k] = 1.0;
		  }
	  }
}



template <int dim>
void Solid<dim>::run()
{
	Viscoelasticity::VEM<dim> material(parameter.lambda, parameter.mu, parameter.eta,  parameter.a_, parameter.c_);

	make_grid();

	system_setup();

	Viscoelasticity_qp_data qp_data(triangulation.n_cells(),std::vector<Viscoelasticity::Data<dim>>(n_q_points));

	reset_viscoelasticity(qp_data);

	output_results(qp_data,material);

	time = 0.0;

	times_and_names.clear();

	for (current_load_step=1; current_load_step <= parameter.N_load_steps ; current_load_step++)
	{
		solution_delta = 0.0;

		for (unsigned int i = 0; i < triangulation.n_cells(); ++i)
			for (unsigned int j = 0; j < n_q_points; ++j)
			{
				for (unsigned int k = 0; k < dim; ++k)
				{
					qp_data[i][j].Fv_A_last_timestep[k][k] = qp_data[i][j].Fv_A_0[k][k];
			        qp_data[i][j].F0_last_timestep[k][k] = qp_data[i][j].F0[k][k];
				}
			}
		solve_load_step_NR(qp_data, material, solution_delta);

		solution_n += solution_delta;

		output_results(qp_data,material);

		time += parameter.delta_t;
	}
}



template <int dim>
void Solid<dim>::make_grid()
{
	AssertThrow(dim==3,ExcMessage("make_grid() only works for dim=3"));

	Point<dim> p1(0,-1,0);
	Point<dim> p2(1,1,1);

	const std::vector<unsigned int> repetitions{1,2,1};

	GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,p1,p2);

	for (auto cell : triangulation.active_cell_iterators())
		if (cell->at_boundary())
			for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
				if (cell->face(f)->at_boundary())
				{
					const Point<dim> face_center = cell->face(f)->center();

					// face with y = -1
					if (face_center[1] == -1)
						cell->face(f)->set_boundary_id(1);

					// face with y = 1
					else if (face_center[1] == 1)
						cell->face(f)->set_boundary_id(2);

					// face with x = -1
					else if (face_center[0] == 0)
						cell->face(f)->set_boundary_id(3);

					// face with x = 1
					else if (face_center[0] == 1)
						cell->face(f)->set_boundary_id(4);

					// face with z = -1
					else if (face_center[2] == 0)
						cell->face(f)->set_boundary_id(5);

					// face with z = 1
					else if (face_center[2] == 1)
						cell->face(f)->set_boundary_id(6);
				}
	//triangulation.refine_global(2);

	GridOut grid_out;
	std::ofstream output("mesh.vtu");
	grid_out.write_vtu(triangulation,output);
}



template <int dim>
void Solid<dim>::system_setup()
{
	dof_handler.distribute_dofs(fe);

	DoFRenumbering::Cuthill_McKee(dof_handler);
	
	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler,constraints);
	constraints.close();
	
	std::cout << "Triangulation:"
			  << "\n\t Number of active cells: " << triangulation.n_active_cells()
			  << "\n\t Number of degrees of freedom: " << dof_handler.n_dofs()
			  << std::endl;

	DynamicSparsityPattern dsp(dof_handler.n_dofs(),dof_handler.n_dofs());

	DoFTools::make_sparsity_pattern(dof_handler,dsp,constraints,true);

	sparsity_pattern.copy_from(dsp);

	tangent_matrix.reinit(sparsity_pattern);

	system_rhs.reinit(dof_handler.n_dofs());

	solution_delta.reinit(dof_handler.n_dofs());

	solution_n.reinit(dof_handler.n_dofs());
}

template <int dim>
void Solid<dim>::assemble_system(Viscoelasticity_qp_data &qp_data,
								 Viscoelasticity::VEM<dim> &material)
{
	// Create FEValues named "fe_v" with the fe-element "fe" and the quadrature rule "qf_cell"
	// we further need information about the values, the gradients and the weights
	// at every quadrature point. You find further information in
	// https://www.dealii.org/9.0.0/doxygen/deal.II/classFEValuesViews_1_1Vector.html#a9e2686feec1a56451b674431b63c92d1


	// create "cell_matrix" and define its size and call the FullMatrix<double> "cell_matrix".
	// The size equals "dofs_per_cell" x "dofs_per_cell"

	// create "cell_rhs" and define its size and call the Vector<double> "cell_rhs"
	// The size equals "dofs_per_cell".

	// this vector stores the global dof indices of the local dof indices
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	// This is the solution of the last iteration.
	Vector<double> current_solution = get_total_solution(this->solution_delta);

	// cell_counter. We need it to extract cell-related quantities
	unsigned int cell_counter = 0;

	// start loop over all cells. Therefore we use dof_handler
	for (auto cell : dof_handler.active_cell_iterators())
	{
		// TODO initialise "cell_matrix" and "cell_rhs"

		// TODO initialise the FEValues "fe_v" for every cell with the command reinit(cell);

		std::vector<Tensor<2,dim>> solution_grads_u(n_q_points);

		// TODO extract solution_gradients at each quadrature point. Therefore use get_function_gradients
		// from FEValues class. Have again a look at
		// https://www.dealii.org/9.0.0/doxygen/deal.II/classFEValuesViews_1_1Vector.html#a9e2686feec1a56451b674431b63c92d1

		// here, we fill "local_dof_indices" with the corresponding global dof indices
		cell->get_dof_indices(local_dof_indices);

		// loop over all quadrature points
		for(unsigned int qp=0; qp<n_q_points;++qp)
		{
			// initialise DeformationGradient
			Tensor<2,dim> DeformationGradient;

			// TODO calculate deformation_gradient

			// here, we perform what is nescessary for viscoelasticity
			material.time_integration(time,qp_data[cell_counter][qp],DeformationGradient,false,parameter.delta_t);
			// we then extract the first Piola-Kirchhoff stress
			Tensor<2,dim> P = material.get_P(DeformationGradient,qp_data[cell_counter][qp]);
            // and the tangent
			Tensor<4,dim> dP_dF = material.get_dP_dF(DeformationGradient,qp_data[cell_counter][qp],time);

			// Initialise mapped weight of quadrature point
			const double JxW;

			// TODO extract "JxW" from "fe_v"

			for(unsigned int i=0; i<dofs_per_cell; ++i)
			{
				// initialise gradient of shape function i
				Tensor <2,dim> shape_grad_i;

				// TODO obtain "shape_grad_i"

				// TODO fill values of "cell_rhs"
				// cell_rhs(i)-=...

				for(unsigned int j=0; j<dofs_per_cell; ++j)
				{
					// initialise gradient of shape function j
					Tensor <2,dim> shape_grad_j;

					// TODO obtain "shape_grad_j"

					// TODO fill values of system_matrix"
					// cell_matrix(i,j)+=...
				}

			}
		}

		// TODO Assemble local to global, taking into account all constraints.

		++cell_counter;
	}
}



template <int dim>
void Solid<dim>::make_constraints(const int &it_nr)
{
  std::cout << " CST " << std::flush;

  if (it_nr >= 2)
	  return;

  constraints.clear();

  const bool inhomogeneous_dbc = (it_nr == 0);

  const FEValuesExtractors::Scalar y_displacement(1);

  const FEValuesExtractors::Scalar x_displacement(0);

  const FEValuesExtractors::Scalar z_displacement(2);

  // symmetry condition
  VectorTools::interpolate_boundary_values(dof_handler,
                                           3,
										   ZeroFunction<dim>(dim),
										   constraints,
										   fe.component_mask(x_displacement));

  // symmetry condition
  VectorTools::interpolate_boundary_values(dof_handler,
                                           5,
										   ZeroFunction<dim>(dim),
										   constraints,
										   fe.component_mask(z_displacement));

  if (inhomogeneous_dbc == true)
  {
	  VectorTools::interpolate_boundary_values(dof_handler,
			  	  	  	  	  	  	  	  	   1,
											   ConstantFunction<dim>(-parameter.step_increment,dim),
											   constraints,
											   fe.component_mask(y_displacement));

	  VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
											   ConstantFunction<dim>(parameter.step_increment,dim),
											   constraints,
											   fe.component_mask(y_displacement));
  }
  else
  {
	  VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
											   ZeroFunction<dim>(dim),
											   constraints,
											   fe.component_mask(y_displacement));

	  VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
											   ZeroFunction<dim>(dim),
											   constraints,
											   fe.component_mask(y_displacement));
  }
  constraints.close();
}


template <int dim>
void Solid<dim>::solve_load_step_NR(Viscoelasticity_qp_data &qp_data,
									Viscoelasticity::VEM<dim> &material,
									Vector<double> &solution_delta)
{
	// Vector containing the solution of the linear system.
	Vector<double> newton_update(dof_handler.n_dofs());

	// print some information for output in console
    print_conv_header();

    // TODO make a loop from 0 to "max_number_of_iterations". Call the counting variable
    // "newton_iteration"
    // for (unsigned int newton_iteration .....)
    {
    	// TODO reinitialise "tangent_matrix" and "system_rhs". So set them equal to zero

    	// TODO Call function "make_constraints". The input variable is the current "newton_iteration"

    	// TODO Call function "assemble_system". Its input variables are the quadrature point history
    	// "qp_data" and the material properties "material". They also appear as variables in
    	// the current function.

    	// TODO Check if the Newton-scheme already converged. Therefore, compare the l2_norm() of the
    	// "system_rhs" with the required tolerance, obtained with "parameter.tolerance_residual". In addition
    	// the current "newton_iteration" shall be gerater 0.
    	{
    		// if converged
    		std::cout << " CONVERGED! " << std::endl;
    		print_conv_footer();
    		break;
    	}

    	// Solve linear system of equation.
    	const std::pair<unsigned int,double> lin_solver_output = solve_linear_system(newton_update);

    	// TODO update "solution_delta" with the incremental solution

    	// some informations about the iteration are printed
    	std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(25)
    			  << std::scientific << lin_solver_output.first << "  "
				  << lin_solver_output.second << "  " << system_rhs.l2_norm()
				  << "  " << std::endl;
    }
}



template <int dim>
void Solid<dim>::print_conv_header()
{
	std::cout << std::endl << std::endl << "Step " << current_load_step << " out of "
			  << parameter.N_load_steps << std::endl;

	const unsigned int l_width = 90;
	for (unsigned int i = 0; i < l_width; ++i)
		std::cout << "_";

	std::cout << std::endl;

	std::cout << "           SOLVER STEP            "
				<< " |  LIN_IT   LIN_RES    RES_NORM    "
				<< std::endl;

	for (unsigned int i = 0; i < l_width; ++i)
		std::cout << "_";

	std::cout << std::endl;
}



template <int dim>
void Solid<dim>::print_conv_footer()
{
	const unsigned int l_width = 90;
	for (unsigned int i = 0; i < l_width; ++i)
		std::cout << "_";

	std::cout << std::endl;

//	std::cout << "Norm of residuum:\t\t"
//			  << error_residual_norm.u << std::endl
//			  << std::endl;
}



template <int dim>
void Solid<dim>::get_error_residual(Errors &error_residual)
{
	Vector<double> error_res(dof_handler.n_dofs());

	for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
		if (!constraints.is_constrained(i))
			error_res(i) = system_rhs(i);

	error_residual.u = error_res.l2_norm();
}



template <int dim>
Vector<double>
Solid<dim>::get_total_solution(const Vector<double> &solution_delta) const
{
	Vector<double> solution_total(solution_n);

	solution_total += solution_delta;

	return solution_total;
}



template <int dim>
std::pair<unsigned int, double>
Solid<dim>::solve_linear_system(Vector<double> &newton_update)
{
	const bool it_solver=false;
	unsigned int lin_it = 0;
	double lin_res = 0.0;
	newton_update=0;
	std::cout << " SLV " << std::flush;
	if (it_solver)
	{
		const int solver_its = tangent_matrix.m();
		const double tol_sol = 1e-12;
		SolverControl solver_control(solver_its, tol_sol);
		SolverGMRES<Vector<double> > solver_GMRES(solver_control);
		solver_GMRES.solve(tangent_matrix,
						   newton_update,
						   system_rhs,
						   PreconditionIdentity());

		lin_it = solver_control.last_step();
		lin_res = solver_control.last_value();
	}
	else
	{
		SparseDirectUMFPACK direct_solver;
		direct_solver.initialize(tangent_matrix);
		direct_solver.vmult(newton_update,system_rhs);

		lin_it = 1;
	}
	constraints.distribute(newton_update);
	return std::make_pair(lin_it,lin_res);
}

//template <int dim>
//std::pair<unsigned int, double>
//Solid<dim>::solve_linear_system(Vector<double> &newton_update)
//{
//	// "it_solver" defines, if an iterative solver shall be used or not
//	const bool it_solver=false;
//
//	// initialise two values evaluating the iterative solver
//	// "lin_it" defines how many iterations the iterative solver needed
//	// "lin_res" defines the residual, when a solution is obtained
//	unsigned int lin_it = 0;
//	double lin_res = 0.0;
//
//	// TODO initialise "newton_update"
//
//	// output
//	std::cout << " SLV " << std::flush;
//
//	if (it_solver)
//	{
//		// TODO create an iterative solver (GMR), where the system matrix is
//		// called "tangent_matrix" and the residual is called "system_rhs".
//		// Both are global variables
//
//		// TODO extract iteration and residual, of converged iteration
//
//		// You find more informations also on:
//		// https://www.dealii.org/9.0.0/doxygen/deal.II/classSolver.html
//	}
//	else
//	{
//		// TODO create a direct solver, where the system matrix is called
//		//"tangent_matrix" and the residual is called "system_rhs". Both are
//		// global variables.
//
//		// a direct solver only requires one iteration
//		lin_it = 1;
//	}
//
//	// distribute constraints
//	// TODO distribute constraints to the solution
//
//	return std::make_pair(lin_it,lin_res);
//}



template <int dim>
void Solid<dim>::output_results(Viscoelasticity_qp_data &/*viscoplasticity_gauss_point_data*/,
								Viscoelasticity::VEM<dim> &/*material*/)
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim,DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim,"u");

    data_out.add_data_vector(solution_n,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    data_out.build_patches();

    std::string filename = "solution-" + Utilities::to_string(current_load_step) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    times_and_names.push_back(std::pair<double,std::string>(time,filename));
    std::ofstream pvd_output("solution.pvd");
    DataOutBase::write_pvd_record(pvd_output,times_and_names);
}
//--------------------------------------------------------------------------



int main ()
{
	using namespace dealii;

	const unsigned int dim=3;

	try
    {
		deallog.depth_console(1);

		std::string parameterfilename = "../Parameters.prm";
	  
		Solid<dim> solid_xd(parameterfilename);
		solid_xd.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
