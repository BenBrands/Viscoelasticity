#ifndef NEOHOOKEANMATERIAL_H
#define NEOHOOKEANMATERIAL_H

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <iostream>

using namespace dealii;



namespace StandardTensors
{
 	 template<int dim>
 	 SymmetricTensor<2,dim> I() {return unit_symmetric_tensor<dim>();}

 	 template<int dim>
 	 SymmetricTensor<4,dim> IxI() {return outer_product(I<dim>(),I<dim>());}

 	 template<int dim>
 	 SymmetricTensor<4,dim> II() {return identity_tensor<dim>();}
}



namespace StrainMeasures
{
	template <int dim>
	SymmetricTensor<2,dim> get_LeftCauchyGreenTensor(const Tensor<2,dim> &F)
	{
		SymmetricTensor<2,dim> LeftCauchyGreenTensor;
    
		LeftCauchyGreenTensor = symmetrize(F * transpose(F));

		return LeftCauchyGreenTensor;
	}
}



template<int dim>
class NeoHookeanMaterial 
{
  	  public:

		NeoHookeanMaterial(double mu,double lambda);

		~NeoHookeanMaterial(){}
    
		double get_J(const Tensor<2,dim> &F);

		Tensor<2,dim> get_P(const Tensor<2,dim> &F);

		SymmetricTensor<2,dim> get_sigma(const Tensor<2,dim> &F);

		Tensor<4,dim> get_dP_dF(const Tensor<2,dim> &F);


  	  private:

		const double mu;

		const double lambda;
};



template <int dim>
NeoHookeanMaterial<dim>::NeoHookeanMaterial(double mu, double lambda)
:
mu(mu),
lambda(lambda)
{
}



template <int dim>
double NeoHookeanMaterial<dim>::get_J(const Tensor<2,dim> &F)
{
	double det_F = determinant(F);

	AssertThrow(det_F>0,ExcMessage("det_F <= 0"));

	return det_F;
}



template <int dim>
SymmetricTensor<2, dim> NeoHookeanMaterial<dim>::get_sigma(const Tensor<2,dim> &F)
{
	Tensor<2,dim> tensor_P = get_P(F);
	double det_F = get_J(F);

	Tensor<2,dim> sigma = 1./det_F * (F * transpose(tensor_P));

	return symmetrize(sigma);
}



template <int dim>
Tensor<2, dim> NeoHookeanMaterial<dim>::get_P(const Tensor<2,dim> &F)
{	
	double det_F = get_J(F);

	Tensor<2,dim> inv_F = invert(F);

	Tensor<2,dim> tensor_P = (lambda * log(det_F) - mu) * dealii::transpose(inv_F);
	tensor_P += mu * F;

	return tensor_P;
}



template <int dim>
Tensor<4,dim> NeoHookeanMaterial<dim>::get_dP_dF(const Tensor<2,dim> &F)
{
	double factor = lambda * log(get_J(F)) - mu;

	Tensor<4,dim> tensor_dP_dF;
	Tensor<2,dim> inv_F = invert(F);

	for (unsigned int i=0; i<dim; ++i)
		for (unsigned int j=0; j<dim; ++j)
			for (unsigned int k=0; k<dim; ++k)
				for (unsigned int l=0; l<dim; ++l)
				{
					if (i==k && j==l) tensor_dP_dF[i][j][k][l] += mu;
					tensor_dP_dF[i][j][k][l] += lambda * inv_F[l][k] * inv_F[j][i];
					tensor_dP_dF[i][j][k][l] -= factor * inv_F[l][i] * inv_F[j][k];
				}
	return tensor_dP_dF;
}



#endif 
