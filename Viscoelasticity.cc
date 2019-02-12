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
#include <deal.II/lac/constraint_matrix.h>
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

	ConstraintMatrix			       constraints;

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
	std::cout << " Assembly " << std::flush;

	FEValues<dim> fe_v(fe,
					   qf_cell,
					   update_values| update_gradients| update_JxW_values);

	FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);

	Vector<double> cell_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	Vector<double> current_solution = get_total_solution(this->solution_delta);

	bool flag = parameter.print_debug;

	unsigned int cell_counter = 0;

	for (auto cell : dof_handler.active_cell_iterators())
	{
		cell_matrix=0.0;

		cell_rhs=0.0;

		fe_v.reinit(cell);

		std::vector<Tensor<2,dim>> solution_grads_u(n_q_points);

		fe_v[u_fe].get_function_gradients(current_solution,solution_grads_u);

		cell->get_dof_indices(local_dof_indices);

		for(unsigned int qp=0; qp<n_q_points;++qp)
		{
			Tensor<2,dim> DeformationGradient =  Auxiliary_Functions::I<dim>() + solution_grads_u[qp];

			material.time_integration(time,qp_data[cell_counter][qp],DeformationGradient,flag,parameter.delta_t);

			Tensor<2,dim> P = material.get_P(DeformationGradient,qp_data[cell_counter][qp]);

			Tensor<4,dim> dP_dF = material.get_dP_dF(DeformationGradient,qp_data[cell_counter][qp],time);

			if (flag)
			{
				std::cout << "  F: " << std::fixed << std::setprecision(10) << DeformationGradient << std::endl;
				std::cout << "  current_solution: " << std::fixed << std::setprecision(10) << current_solution << std::endl;
			}

			const double JxW = fe_v.JxW(qp);

			for(unsigned int i=0; i<dofs_per_cell; ++i)
			{
				const unsigned int component_i = fe.system_to_component_index(i).first;

				Tensor<1,dim> shape_grad_i_vec = fe_v.shape_grad(i,qp);

				cell_rhs(i) -= (P * shape_grad_i_vec)[component_i] * JxW;

				for(unsigned int j=0; j<dofs_per_cell; ++j)
				{
					const unsigned int component_j = fe.system_to_component_index(j).first;

					Tensor<1,dim> shape_grad_j_vec = fe_v.shape_grad(j,qp);

					Tensor<2,dim> K = Auxiliary_Functions::tangent_multiplication(dP_dF,shape_grad_i_vec,shape_grad_j_vec);

					cell_matrix(i,j) += K[component_i][component_j] * JxW;
				}
				flag = false;
			}
		}
		constraints.distribute_local_to_global(cell_matrix,
                                           	   cell_rhs,
											   local_dof_indices,
											   tangent_matrix,
											   system_rhs,
											   false);

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
	Vector<double> newton_update(dof_handler.n_dofs());

    error_residual.reset();

    error_residual_0.reset();

    error_residual_norm.reset();

    print_conv_header();

    for (unsigned int newton_iteration = 0; newton_iteration < max_number_newton_iterations; ++newton_iteration)
    {
    	std::cout << " " << std::setw(2) << newton_iteration << " " << std::flush;

    	tangent_matrix = 0.0;
    	system_rhs = 0.0;

    	make_constraints(newton_iteration);

    	assemble_system(qp_data, material);

    	get_error_residual(error_residual);

    	if (newton_iteration == 0)
    		error_residual_0 = error_residual;

    	error_residual_norm = error_residual;
    	error_residual_norm.normalise(error_residual_0);

    	if (newton_iteration > 0 && error_residual.u <= parameter.tolerance_residual)
    	{
    		std::cout << " CONVERGED! " << std::endl;
    		print_conv_footer();
    		break;
    	}

    	const std::pair<unsigned int,double> lin_solver_output = solve_linear_system(newton_update);

    	solution_delta += newton_update;

    	std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(17)
    			  << std::scientific << lin_solver_output.first << "  "
				  << lin_solver_output.second << "  " << error_residual.u
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

	std::cout << "Norm of residuum:\t\t"
			  << error_residual_norm.u << std::endl
			  << std::endl;
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
