#include <iostream>

#include <guts_utils/Grid.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;
using Grid = guts_utils::Grid;
using Point = guts_utils::Grid::Point;
using Cell = guts_utils::Cell;

PYBIND11_MODULE(grid, g)
{
  g.doc() = "Grid and raytracing python bindings";

  py::class_<Cell>(g, "Cell")
    .def(py::init<unsigned int, unsigned int>())
    .def(py::init<unsigned int, unsigned int>())
    .def_readwrite("row", &Cell::row)
    .def_readwrite("col", &Cell::col);

  py::class_<Grid>(g, "Grid")
    .def(py::init<Point, unsigned int, unsigned int, float>())
    .def("set", &Grid::set)
    .def("point_to_cell", &Grid::pointToCell)
    .def("point_to_index", &Grid::pointToIndex)
    .def("cell_to_point", &Grid::cellToPoint)
    .def("index_to_cell", &Grid::indexToCell)
    .def("cell_to_index", &Grid::cellToIndex)
    .def("in_grid", py::overload_cast<Point>(&Grid::inGrid, py::const_))
    .def("in_grid", py::overload_cast<Cell>(&Grid::inGrid, py::const_))
    .def("in_grid", py::overload_cast<unsigned int>(&Grid::inGrid, py::const_))
    .def("update_row", &Grid::update_row)
    .def("update_col", &Grid::update_col)
    .def("raytrace", &Grid::raytrace)
    .def_readwrite("width", &Grid::width)
    .def_readwrite("height", &Grid::height)
    .def_readwrite("resolution", &Grid::resolution)
    .def_readwrite("origin", &Grid::origin)
    .def_readwrite("data", &Grid::data)
    ;
}
