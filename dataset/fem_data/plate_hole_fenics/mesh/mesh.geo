// Gmsh project created on Wed Apr 08 16:48:07 2026
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0.1, 0, 0, 1.0};
//+
Point(3) = {1, 0, 0, 1.0};
//+
Point(4) = {1, 1, 0, 1.0};
//+
Point(5) = {0, 1, 0, 1.0};
//+
Point(6) = {0, 0.1, 0, 1.0};
//+
Line(1) = {2, 3};
//+
Line(2) = {3, 4};
//+
Line(3) = {4, 5};
//+
Line(4) = {5, 6};
//+
Circle(5) = {6, 1, 2};
//+
Curve Loop(1) = {3, 4, 5, 1, 2};
//+
Plane Surface(1) = {1};
//+
Physical Curve("arch", 6) = {5};
//+
Physical Curve("left", 7) = {4};
//+
Physical Curve("left", 7) += {4};
//+
Physical Curve("top", 8) = {3};
//+
Physical Curve("right", 9) = {2};
//+
Physical Curve("bottom", 10) = {1};
//+
Show "*";
//+
Hide {
  Point{1}; Surface{1}; 
}
//+
Show "*";
//+
Physical Surface("surface", 11) = {1};
//+
Show "*";
//+
Hide {
  Point{1}; 
}
