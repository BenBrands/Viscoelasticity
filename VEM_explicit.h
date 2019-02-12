#ifndef TNM_H
#define TNM_H


#include "NeoHookeanMaterial.h"


namespace Auxiliary_Functions
{
	template<int dim>
	Tensor<2,dim> I()
	{
		Tensor<2,dim> I;

		for (unsigned int i = 0; i < dim; ++i)
			I[i][i] = 1.0;

		return I;
	}



	template<int dim>
	Tensor<2,dim> Dev_tensor(const Tensor<2,dim> &A)
	{
		return A - 1.0/3.0 * trace(A) * Auxiliary_Functions::I<dim>();
	}



	template<int dim>
	Tensor<2,dim> tangent_multiplication(const Tensor<4,dim> &T, const Tensor<1,dim> &Nj, const Tensor<1,dim> &Nl)
	{
		Tensor<3,dim> temp;
		Tensor<2,dim> res;

		for (unsigned int i = 0; i < dim; ++i)
		  for (unsigned int j = 0; j < dim; ++j)
			for (unsigned int k = 0; k < dim; ++k)
			  for (unsigned int l = 0; l < dim; ++l)
				temp[i][k][l] += T[i][j][k][l] * Nj[j];

		for (unsigned int i = 0; i < dim; ++i)
		  for (unsigned int k = 0; k < dim; ++k)
			for (unsigned int l = 0; l < dim; ++l)
			  res[i][k] += temp[i][k][l] * Nl[l];

		return res;
	}
}



namespace Viscoelasticity
{
  	  template<int dim>
  	  struct Data
	  {
  		  Tensor<2, dim> Fv_A_0;
  		  Tensor<2, dim> Fv_A_last_timestep;
  		  Tensor<2, dim> F0;
  		  Tensor<2, dim> F0_last_timestep;
	  };

  	  template<int dim>
  	  class VEM
	  {
	  	  private:

		  const double lambda, mu;
		  const double eta;

		  const double a_;
		  const double c_;

		  NeoHookeanMaterial<dim> neo_hooke;

	  	  public:

		  VEM(double lambda, double mu, double eta, double a_, double c_);

		  ~VEM(){};

		  Tensor<2,dim> get_Fv_dot(const Tensor<2,dim> &Fv, const Tensor<2, dim> &F0);

		  void time_integration(const double &t0, Viscoelasticity::Data<dim> &last_step,
				  	  	  	  	const Tensor<2,dim> &F_k, const bool &flag, const double &delta_t);

		  Tensor<2,dim> get_P(const Tensor<2,dim> &F, Viscoelasticity::Data<dim> &last_step);

		  Tensor<4,dim> get_dP_dF(const Tensor<2,dim> &F, Viscoelasticity::Data<dim> &last_step, double &t);

		  Tensor<4,dim> get_dP_dF_C(const Tensor<2,dim> &F);

		  Tensor<4,dim> get_dP_dF_A(const Tensor<2,dim> &F, Viscoelasticity::Data<dim> &last_step);
	  };



  	  template <int dim>
  	  VEM<dim>::VEM(double lambda, double mu, double eta,  double a_, double c_)
	  :
	  lambda(lambda),
	  mu(mu),
	  eta(eta),
	  a_(a_),
	  c_(c_),
	  neo_hooke(lambda,mu)
	  {}



	  template<int dim>
	  Tensor<2,dim> VEM<dim>::get_Fv_dot(const Tensor<2,dim> &Fv, const Tensor<2,dim> &F0)
	  {
		  Tensor<2,dim> FvDot;
		  Tensor<2,dim> F;
		  Tensor<2,dim> Fe;
		  Tensor<2,dim> sigma;

		  F = F0;
		  Fe = F * invert(Fv);

		  sigma = neo_hooke.get_sigma(Fe);
/*		  Tensor<2,dim> deviator_directions = Auxiliary_Functions::Dev_tensor<dim>(sigma);
		  deviator_directions *= 1./deviator_directions.norm();*/

		  FvDot = (1./eta) * invert( Fe ) * Auxiliary_Functions::Dev_tensor<dim>(sigma) * F;

		  return FvDot;
	  }



	  template<int dim>
	  void VEM<dim>::time_integration(const double &/*t0*/, Viscoelasticity::Data<dim> &last_step,
									  const Tensor<2,dim> &F_k, const bool &/*flag*/, const double &delta_t)
	  {
			const Tensor<2,dim> F_v_A = last_step.Fv_A_last_timestep
									   + delta_t * get_Fv_dot(last_step.Fv_A_last_timestep,last_step.F0_last_timestep);

			last_step.Fv_A_0 = F_v_A;
			last_step.F0 = F_k;
	  }



	  template<int dim>
	  Tensor<2,dim> VEM<dim>::get_P(const Tensor<2,dim> &F, Viscoelasticity::Data<dim> &last_step)
	  {
			Tensor<2,dim> Fe_A = F * invert(last_step.Fv_A_0);
			Tensor<2,dim> Fe_C = F;

			Tensor<2,dim> sigma_A = neo_hooke.get_sigma(Fe_A);
			Tensor<2,dim> sigma_C = neo_hooke.get_sigma(Fe_C);

			const double J = determinant(F);
			Tensor<2,dim> transpose_inv_F = transpose(invert(F));

			Tensor<2, dim> P_A, P_C;
			P_A = J * sigma_A * transpose_inv_F;
			P_C = J * sigma_C * transpose_inv_F;

			return a_*P_A + c_*P_C;
	  }



	  template<int dim>
	  Tensor<4,dim> VEM<dim>::get_dP_dF_C(const Tensor<2,dim> &F)
	  {
		  return neo_hooke.get_dP_dF(F);
	  }



	  template<int dim>
	  Tensor<4,dim> VEM<dim>::get_dP_dF_A(const Tensor<2,dim> &F, Viscoelasticity::Data<dim> &last_step)
	  {
		  const Tensor<2,dim> Fv = last_step.Fv_A_0;
		  const Tensor<2,dim> Fe = F * invert(Fv);

		  return neo_hooke.get_dP_dF(Fe);
	  }



	  template<int dim>
	  Tensor<4,dim> VEM<dim>::get_dP_dF(const Tensor<2,dim> &F, Viscoelasticity::Data<dim> &last_step, double &/*t*/)
	  {
		  const Tensor<4,dim> dP_dF_A = get_dP_dF_A(F,last_step);
		  const Tensor<4,dim> dP_dF_C = get_dP_dF_C(F);

		  return a_*dP_dF_A + c_*dP_dF_C;
	  }
}

#endif
