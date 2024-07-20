#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <guts_utils/GUTSUtils.h>
#include <guts_utils/Grid.h>

namespace py = pybind11;
namespace gu = guts_utils;
using Grid  = typename gu::Grid;

std::pair<gu::Matrix, gu::Vector>
createDirectionalSensor(gu::GUTS n, const gu::Vector2& curr_pos,
			const gu::Vector2& next_pos, const Grid& map)
{
  gu::Matrix x;
  gu::Vector noise_var;

  n.createDirectionalSensor(curr_pos, next_pos, map, x, noise_var);

  return std::pair<gu::Matrix, gu::Vector>(x, noise_var);
}


PYBIND11_MODULE(guts_utils, m)
{
  m.doc() = "Python binding over guts_utils package";

  py::class_<gu::GUTS>(m, "GUTS")
    .def(py::init<>())
    .def("multivariate_normal", &gu::GUTS::multivariateNormal)
    .def("posterior", &gu::GUTS::posterior)
    .def("create_directional_sensor", createDirectionalSensor)
    .def("loss", &gu::GUTS::loss)
    .def("score", &gu::GUTS::score)
    .def_readwrite("beta_hat", &gu::GUTS::beta_hat_)
    .def_readwrite("gamma", &gu::GUTS::gamma_)
    .def_readwrite("min_eig", &gu::GUTS::min_eig_)
    .def_readwrite("beta_tilde", &gu::GUTS::beta_tilde_)
    .def_readwrite("Gamma", &gu::GUTS::Gamma_)
    .def_readwrite("Sig_beta", &gu::GUTS::Sig_beta_)
    .def_readwrite("ginv_xtsx", &gu::GUTS::ginv_xtsx_)
    .def_readwrite("inv_y_one", &gu::GUTS::inv_y_one_)
    ;
}
