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
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>

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

	ConstraintMatrix		           constraints;

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

	// hyper rectangle
	Point<dim> p1(0,-1,0);
	Point<dim> p2(1,1,1);

	const std::vector<unsigned int> repetitions{3,5,3};

	GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,p1,p2);

//	// hyper cube with cylindrical hole
//	const double inner_radius = 0.25;
//	const double outer_radius = 0.5;
//	const double dimension_in_z = 0.5;
//	unsigned int repetitions_in_z = 2;
//	const bool colorize = false;
//	GridGenerator::hyper_cube_with_cylindrical_hole(triangulation,
//													inner_radius,
//													outer_radius,
//													dimension_in_z,
//													repetitions_in_z,
//													colorize);


	// assign boundary ids
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

	//global refinement
	triangulation.refine_global(2);


	//local refinement
	for (auto cell : triangulation.active_cell_iterators()){
		// based on location of cell_center
		if(cell->center()[2]< 0.5){
			cell->set_refine_flag();
		}
		else{
			cell->set_coarsen_flag();
		}
		// based on material_id
//		if(cell->material_id()==2){
//			cell->set_refine_flag();
//		}
	}
	triangulation.execute_coarsening_and_refinement();

	std::cout<< "Number of active cells: "
			<< triangulation.n_active_cells()
			<< std::endl;
	std::cout<< "Total number of cells: "
			<< triangulation.n_cells()
			<< std::endl;

	GridOut grid_out;
	std::ofstream output("mesh.vtu");
	grid_out.write_vtu(triangulation,output);
	exit(0);
}



template <int dim>
void Solid<dim>::system_setup()
{
	dof_handler.distribute_dofs(fe);
	std::cout<< "Number of degrees of freedom: "
			<< dof_handler.n_dofs()
			<< std::endl;


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

}



template <int dim>
void Solid<dim>::output_results(Viscoelasticity_qp_data &/*viscoplasticity_gauss_point_data*/,
								Viscoelasticity::VEM<dim> &/*material*/)
{

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

