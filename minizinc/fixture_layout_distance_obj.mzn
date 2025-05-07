float: max_dim_fixture;
float: max_half_dim_fixture;
float: max_area_fixture;

float: width;
float: height;
float: available_surface;
int: ntypes;
float: bar_size;
int: max_f;

int: nline_below;
set of int: LINE_BELOW = 1..nline_below;
array[LINE_BELOW] of float: m_below;
array[LINE_BELOW] of float: q_below;

int: nline_above;
set of int: LINE_ABOVE = 1..nline_above;
array[LINE_ABOVE] of float: m_above;
array[LINE_ABOVE] of float: q_above;

int: nline_left;
set of int: LINE_LEFT = 1..nline_left;
array[LINE_LEFT]of float: q_left;

int: nline_right;
set of int: LINE_RIGHT = 1..nline_right;
array[LINE_RIGHT]of float: q_right;


set of int: TYPE = 1..ntypes;
array[TYPE] of float: number;

set of int: FIXTURE = 1..max_f;

enum Dims = {XSize, YSize};

array[TYPE, Dims] of float: dims;
array[TYPE, Dims] of float: half_dims;

array[FIXTURE] of var 0..width: x;
array[FIXTURE] of var 0..height: y;


array[FIXTURE, TYPE] of var 0..1: selected_fixture;
array[FIXTURE] of var 0..1: selected_fixture_array;
var 0..max_f: n =  sum([selected_fixture_array[c] | c in FIXTURE]);

array[FIXTURE] of var 0..max_dim_fixture: fixture_dims_x;
array[FIXTURE] of var 0..max_dim_fixture: fixture_dims_y;
array[FIXTURE] of var 0..max_half_dim_fixture: fixture_half_dims_x;
array[FIXTURE] of var 0..max_half_dim_fixture: fixture_half_dims_y;
array[FIXTURE] of var 0..ntypes: fixture_type;

array[FIXTURE] of var 0..max_area_fixture: fixture_area;
array[FIXTURE] of var 0..width: fixtures_center_x;
array[FIXTURE] of var 0..width: fixtures_center_y;
var float: weighted_cx_sum = sum([fixture_area[c] * fixtures_center_x[c] | c in FIXTURE]);
var float: weighted_cy_sum = sum([fixture_area[c] * fixtures_center_y[c] | c in FIXTURE]);
var 0..width: xG;
var 0..height: yG;


predicate stay_below_line(var float: m, var float: q, var float: x_bl, var float: y_bl, var float: x_size, var float: y_size) = 
             let {
                      var float: x_br = x_bl + x_size,
                      var float: y_br = y_bl,
                      var float: x_tl = x_bl,
                      var float: y_tl = y_bl + y_size,
                      var float: x_tr = x_bl + x_size,
                      var float: y_tr = y_bl + y_size
                  } in
                      y_bl < m * x_bl + q /\
                      y_br < m * x_br + q /\
                      y_tl < m * x_tl + q /\
                      y_tr < m * x_tr + q;
                      
predicate stay_above_line(var float: m, var float: q, var float: x_bl, var float: y_bl, var float: x_size, var float: y_size) = 
             let {
                      var float: x_br = x_bl + x_size,
                      var float: y_br = y_bl,
                      var float: x_tl = x_bl,
                      var float: y_tl = y_bl + y_size,
                      var float: x_tr = x_bl + x_size,
                      var float: y_tr = y_bl + y_size
                  } in
                      y_bl > m * x_bl + q /\
                      y_br > m * x_br + q /\
                      y_tl > m * x_tl + q /\
                      y_tr > m * x_tr + q;
                      
constraint forall(c in FIXTURE)(
            sum([selected_fixture[c, t] | t in TYPE]) <= 1
          );
          
constraint forall(t in TYPE)(
            sum([selected_fixture[c, t] | c in FIXTURE]) <= number[t]
          );
          
constraint forall(c in FIXTURE)(
            selected_fixture_array[c] = sum(t in TYPE)(selected_fixture[c,t])
);

constraint forall (c in FIXTURE, t in TYPE)
            (
            c > n -> selected_fixture[c, t] = 0 
            );
                    
constraint forall(c in FIXTURE)(
  fixture_type[c] = sum([selected_fixture[c, t] * t | t in TYPE]) /\
  fixture_dims_x[c] = sum(t in TYPE)(selected_fixture[c, t] * dims[t, XSize]) /\
  fixture_dims_y[c] = sum(t in TYPE)(selected_fixture[c, t] * dims[t, YSize]) /\ 
  fixture_half_dims_x[c] = sum(t in TYPE)(selected_fixture[c, t] * half_dims[t, XSize]) /\
  fixture_half_dims_y[c] = sum(t in TYPE)(selected_fixture[c, t] * half_dims[t, YSize])
);
                                           
%fit within shape
constraint forall(c in FIXTURE, t in TYPE)(   
                (x[c] + fixture_dims_x[c]) * selected_fixture[c, t] <= width /\
                (y[c] + fixture_dims_y[c]) * selected_fixture[c, t] <= height
           );

constraint forall(c in FIXTURE, l in LINE_LEFT where selected_fixture_array[c] = 1)
           (
              x[c] > q_left[l]
           );
           
constraint forall(c in FIXTURE, t in TYPE, l in LINE_RIGHT where selected_fixture_array[c] = 1)
           (
              x[c] + fixture_dims_x[c] < q_right[l]
           );           
           
constraint forall(c in FIXTURE, l in LINE_BELOW where selected_fixture_array[c] = 1)
            (
             stay_below_line(m_below[l], q_below[l], x[c], y[c], fixture_dims_x[c], fixture_dims_y[c])
            );
            
constraint forall(c in FIXTURE, l in LINE_ABOVE where selected_fixture_array[c]=1)
            (
             stay_above_line(m_above[l], q_above[l], x[c], y[c], fixture_dims_x[c], fixture_dims_y[c])
            );
            

%align fixtures in bar
constraint forall(i, j in FIXTURE where x[i] < x[j]  /\ selected_fixture_array[i] = 1 /\ selected_fixture_array[j] = 1)
           (
               (abs(x[j] - (x[i] + fixture_dims_x[i])) > bar_size \/ x[i] = x[j])
            );


%no overlap  
constraint forall(c1, c2 in FIXTURE where c1 < c2)
           (
            (x[c1] + fixture_dims_x[c1]) * selected_fixture_array[c1] <= x[c2] * selected_fixture_array[c2] \/
            (x[c2] + fixture_dims_x[c2]) * selected_fixture_array[c2] <= x[c1] * selected_fixture_array[c1] \/
            (y[c1] + fixture_dims_y[c1]) * selected_fixture_array[c1] <= y[c2] * selected_fixture_array[c2] \/
            (y[c2] + fixture_dims_y[c2]) * selected_fixture_array[c2] <= y[c1] * selected_fixture_array[c1]
           );
           
                                                                                                                                                                                                                                                         
%symmetry breaking
include "lex_greater.mzn";
constraint forall(i, j in FIXTURE where i < j)(
    (fixture_type[i] = fixture_type[j] -> lex_greater([x[i], y[i]], [x[j], y[j]])) 
);

%compute center of fixtures
constraint forall(c in FIXTURE)
           (
            fixtures_center_x[c] = (x[c] + fixture_half_dims_x[c]) * selected_fixture_array[c]
           );            
           
constraint forall(c in FIXTURE)
           (
            fixtures_center_y[c] = (y[c] + fixture_half_dims_y[c]) * selected_fixture_array[c]
           );

constraint forall(c in FIXTURE)
            (
              fixture_area[c] = fixture_dims_x[c] * fixture_dims_y[c] * selected_fixture_array[c]
            );           

%compute overall center of gravity
constraint xG * sum([fixture_area[c] | c in FIXTURE]) = weighted_cx_sum;
constraint yG * sum([fixture_area[c] | c in FIXTURE]) = weighted_cy_sum;


constraint sum([selected_fixture_array[c] | c in FIXTURE]) > 2;

output [
  "{\n",
  "\"x\": ", show([i | i in x]), ",\n",
  "\"y\": ", show([i | i in y]), ",\n",
  "\"fixtures_center_x\": ", show([i | i in fixtures_center_x]), ",\n",
  "\"fixtures_center_y\": ", show([i | i in fixtures_center_y]), ",\n",
  "\"selected_fixture\": ", show(selected_fixture_array), ",\n",
  "\"fixture_type\": ", show(fixture_type), ",\n",
  "\"x_g\": ", show(xG), ",\n",
  "\"y_g\": ", show(yG), ",\n",
  "\"objective_value\": ", show(_objective), "\n",
  "}\n"
];

solve maximize sum([abs(xG - fixtures_center_x[c]) * selected_fixture_array[c]| c in FIXTURE]) + sum([abs(yG - fixtures_center_y[c]) * selected_fixture_array[c]| c in FIXTURE])
