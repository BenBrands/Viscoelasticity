#ifndef PARAMETER_H
#define PARAMETER_H

#include <deal.II/base/parameter_handler.h>



using namespace dealii;



namespace Parameter
{

	struct GeneralParameters
	{
		GeneralParameters(const std::string &input_filename);

		bool print_debug;
		double tolerance_residual;
		double step_increment;
		unsigned int N_load_steps;
		unsigned int max_steps_multiplier;
		double delta_t;
		double a_;
		double c_;

		double lambda;
		double mu;
		double eta;
    
		void declare_parameters(ParameterHandler &prm);

		void parse_parameters(ParameterHandler &prm);
	};



  GeneralParameters::GeneralParameters(const std::string &input_filename)
  {
    ParameterHandler prm;
    declare_parameters(prm);
    prm.parse_input(input_filename);
    parse_parameters(prm);
  }



  void GeneralParameters::declare_parameters(ParameterHandler &prm)
  {
	  prm.enter_subsection ("General");
	  {
		  prm.declare_entry("Print for debugging","false",
                          	Patterns::Bool(),
							"Print values for debugging");

		  prm.declare_entry("Tolerance residual","1e-6",
				  	  	    Patterns::Double(),
							"The tolerance w.r.t the normalised residual norm");

		  prm.declare_entry("Displacement step increment","1.0e-1",
                        	Patterns::Double(),
							"Displacement step increment");

		  prm.declare_entry("Total number of load steps","10",
							Patterns::Integer(),
							"Total number of load steps");

		  prm.declare_entry("Multiplier NRsteps","2",
							Patterns::Integer(),
							"Multiplier NRsteps");

		  prm.declare_entry("Time increment","0.5",
							Patterns::Double(),
							"Time increment");

		  prm.declare_entry("a","1.0",
							Patterns::Double(),
							"a");

		  prm.declare_entry("c","1.0",
							Patterns::Double(),
							"c");

		  prm.declare_entry("lambda","290.0",
							Patterns::Double(),
							"First Lame Parameter");

		  prm.declare_entry("mu","2000.0",
							Patterns::Double(),
							"Second Lame Parameter");

		  prm.declare_entry("eta","2000.0",
							Patterns::Double(),
							"Viscosity");
	  }
	  prm.leave_subsection ();
  }



  void GeneralParameters::parse_parameters(ParameterHandler &prm)
  {
	  prm.enter_subsection("General");
	  {
		  print_debug=prm.get_bool("Print for debugging");

		  tolerance_residual=prm.get_double("Tolerance residual");

		  step_increment=prm.get_double("Displacement step increment");

		  N_load_steps=prm.get_integer("Total number of load steps");

		  max_steps_multiplier=prm.get_integer("Multiplier NRsteps");

		  delta_t=prm.get_double("Time increment");

		  a_=prm.get_double("a");

		  c_=prm.get_double("c");

		  lambda=prm.get_double("lambda");

		  mu=prm.get_double("mu");

		  eta=prm.get_double("eta");
	  }
	  prm.leave_subsection();
  }

}//END namespace
#endif 
