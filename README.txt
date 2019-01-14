sparseSVM

*****************************************************************************************************************
* author: Jordan Frécon  											*
* institution: Univ Lyon, Ens de Lyon, Univ Claude Bernard, CNRS, Laboratoire de Physique, F-69342 Lyon, France *
* date: March 03 2017     	              									*
* License CeCILL-B                                    								*
*****************************************************************************************************************


*********************************************************
* RECOMMENDATIONS:                                   	*
* This toolbox is designed to work with Matlab 2015.a   *
*********************************************************

------------------------------------------------------------------------------------------------------------------------
DESCRIPTION:
Sparse SVM for imbalanced classes.
- The sparse SVM objective function considered here is composed of the square hinge loss function as the data fidelity term and the l1 norm as the penalization term.
- The data fidelity term is further split into 2 terms in order to account for the possible imbalanced size between the two classes.

------------------------------------------------------------------------------------------------------------------------
SPECIFICATIONS for using sparseSVM:

One demo file ‘demo_sparseSVM.m’ is proposed.
The main function is ‘sparseSVM’.

------------------------------------------------------------------------------------------------------------------------
RELATED PUBLICATION:

# J. Spilka, J. Frecon, R.F. Leonarduzzi, N. Pustelnik, P. Abry, and M. Doret,
Sparse Support Vector Machine for Intrapartum Fetal Heart Rate Classification, 
Accepted to IEEE Journal of Biomedical and Health Informatics, 2016.

------------------------------------------------------------------------------------------------------------------------