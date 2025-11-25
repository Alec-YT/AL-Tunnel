// Gmsh project created on Mon Mar 17 14:55:21 2025
SetFactory("OpenCASCADE");
//+
Point(1) = {-5, 0, 0, 0.1};
//+
Point(2) = {0, 5, 0, 0.1};
//+
Point(3) = {5, 0, 0, 0.1};
//+
Point(4) = {0, -5, 0, 0.1};
//+
Point(5) = {0, 0, 0, 0.1};
//+
Point(6) = {-150, 0, 0, 10};
//+
Point(7) = {-150, 150, 0, 10};
//+
Point(8) = {0, 150, 0, 10};
//+
Point(9) = {150, 150, 0, 10};
//+
Point(10) = {150, 0, 0, 10};
//+
Point(11) = {150, -150, 0, 10};
//+
Point(12) = {0, -150, 0, 10};
//+
Point(13) = {-150, -150, 0, 10};
//+
Circle(1) = {1, 5, 2};
//+
Circle(2) = {2, 5, 3};
//+
Circle(3) = {3, 5, 4};
//+
Circle(4) = {4, 5, 1};
//+
Line(5) = {6, 7};
//+
Line(6) = {7, 8};
//+
Line(7) = {8, 9};
//+
Line(8) = {9, 10};
//+
Line(9) = {10, 11};
//+
Line(10) = {11, 12};
//+
Line(11) = {12, 13};
//+
Line(12) = {13, 6};
//+
Curve Loop(1) = {5, 6, 7, 8, 9, 10, 11, 12};
//+
Curve Loop(2) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1, 2};
//+
Physical Surface("rock", 13) = {1};
//+
Physical Curve("top", 14) = {6, 7};
//+
Physical Curve("right", 15) = {8, 9};
//+
Physical Curve("bottom", 16) = {10, 11};
//+
Physical Curve("left", 17) = {12, 5};
//+
Physical Curve("wall", 18) = {1, 2, 3, 4};
//+
Physical Point("ml", 19) = {6};
//+
Physical Point("mt", 20) = {8};
//+
Physical Point("mr", 21) = {10};
//+
Physical Point("mb", 22) = {12};
