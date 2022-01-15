/* Author: Jonathan Siegel

Tests the improved convergence rate of the orthogonal greedy algorithm on a simple
problem using the dictionary of Heaviside functions in 2d on the unit square.*/

#include<algorithm>
#include<cstdio>
#include<cstdlib>
#include<fstream>
#include<iostream>
#include<cmath>
#include<vector>
#include<set>

using std::vector;
using std::rand;
using std::sort;
using std::set;
using std::abs;
using std::ofstream;

int sample_point_count = 5000;

// Calculates the dictionary element (indicator function of a hyperplane) 
// which maximizes the inner product with function vals.
struct index_comp {
  vector<vector<double> >* sample_points;
  bool operator() (int i, int j) {return (*sample_points)[i][0] < (*sample_points)[j][0];}
};

struct comp_class_thetas {
  vector<vector<double> >* sample_points;
  vector<int>* indices;
  double* theta_val;

  double theta(vector<double>& p1, vector<double>& p2) {
    double x = p2[0] - p1[0];
    double y = p2[1] - p1[1];
    double angle = 0.5 * M_PI + asin(y / sqrt(x*x + y*y));
    if (x < 0) angle = 2.0 * M_PI - angle;
    return angle;
  }

  bool operator() (int i, int j) {
    double theta_1 = theta((*sample_points)[(*indices)[i]], (*sample_points)[(*indices)[i+1]]);
    double theta_2 = theta((*sample_points)[(*indices)[j]], (*sample_points)[(*indices)[j+1]]);
    theta_1 = theta_1 - *theta_val;
    theta_2 = theta_2 - *theta_val;
    if (theta_1 < 0) theta_1 += 2.0 * M_PI;
    if (theta_2 < 0) theta_2 += 2.0 * M_PI;
    if (theta_1 == theta_2) return (i < j);
    return (theta_1 < theta_2);
  }
};

struct comp_class_vals {
  vector<double>* values;

  bool operator() (int i, int j) {
    if (abs((*values)[i]) == abs((*values)[j])) return (i < j);
    return (abs((*values)[i]) < abs((*values)[j]));
  }
};

void find_optimal_hyperplane(vector<vector<double> > sample_points, vector<double> function_vals, 
                             double& w_1, double& w_2, double& b) {
  vector<int> indices;
  for (int i = 0; i < sample_points.size(); i++) indices.push_back(i);
  index_comp index_comparison = {&sample_points};
  sort(indices.begin(), indices.end(), index_comparison);
  
  // Sorts consecutive points by angle which swaps their order.
  double theta_base = 0;
  comp_class_thetas theta_comparison = {&sample_points, &indices, &theta_base};
  set<int,comp_class_thetas> sorted_points(theta_comparison);
  for (int i = 0; i < sample_points.size() - 1; i++) {
    sorted_points.insert(i);
  }

  // Calculate and sort partial sums of function values.
  vector<double> function_sums;
  double partial_sum = 0;
  for (int i = 0; i < indices.size(); ++i) {
    partial_sum += function_vals[indices[i]];
    function_sums.push_back(partial_sum);
  }
  comp_class_vals vals_comparison = {&function_sums};
  set<int,comp_class_vals> sorted_vals(vals_comparison);
  for (int i = 0; i < indices.size(); i++) sorted_vals.insert(i);

  // Set initial maximum values and optimal theta and b.
  int max_index = *sorted_vals.rbegin();
  double max_value = abs(function_sums[max_index]);
  w_1 = 1.0;
  w_2 = 0;
  b = sample_points[indices[max_index]][0];
  double theta_prev = 0;

  // Recurse through angles which swap the order of elements.
  while (theta_comparison.theta(sample_points[indices[*sorted_points.begin()]],
                                sample_points[indices[*sorted_points.begin() + 1]]) > theta_prev) {
    int swapped_index = *sorted_points.begin();
    double theta_new = theta_comparison.theta(sample_points[indices[*sorted_points.begin()]],
                                sample_points[indices[*sorted_points.begin() + 1]]);
    
    // Remove modified intervals from the set.
    sorted_points.erase(swapped_index);
    if (swapped_index > 0) sorted_points.erase(swapped_index - 1);
    if (swapped_index < indices.size() - 2) sorted_points.erase(swapped_index + 1);
    
    // Swap the order of the indices as theta moves.
    int temp = indices[swapped_index];
    indices[swapped_index] = indices[swapped_index + 1];
    indices[swapped_index + 1] = temp;

    // Update the comparison theta val and add back the updated indices.
    theta_base = theta_new;
    sorted_points.insert(swapped_index);
    if (swapped_index > 0) sorted_points.insert(swapped_index - 1);
    if (swapped_index < indices.size() - 2) sorted_points.insert(swapped_index + 1);

    // Update the ordered partial sums.
    sorted_vals.erase(swapped_index);
    function_sums[swapped_index] += (function_vals[indices[swapped_index]] - function_vals[indices[swapped_index + 1]]);
    sorted_vals.insert(swapped_index);

    // Compare new maximum with old maximum and update w1, w2, and b accordingly.
    int new_max_index = *sorted_vals.rbegin();
    if (abs(function_sums[new_max_index]) > max_value) {
      max_value = abs(function_sums[new_max_index]);
      double theta_max = 0.5 * (theta_new + theta_comparison.theta(sample_points[indices[*sorted_points.begin()]],
                                sample_points[indices[*sorted_points.begin() + 1]]));
      // Deal with an edge case when averaging angles.
      if (theta_new - theta_comparison.theta(sample_points[indices[*sorted_points.begin()]],
                                sample_points[indices[*sorted_points.begin() + 1]]) >= M_PI) theta_max += M_PI;
      w_1 = -1.0 * cos(theta_max);
      w_2 = -1.0 * sin(theta_max);
      b = -1.0 * sample_points[indices[new_max_index]][0] * w_1 - sample_points[indices[new_max_index]][1] * w_2;
    }

    theta_prev = theta_new;
  }
}

int main() {
  // Generate sample points
  vector<vector<double> > sample_points (sample_point_count, vector<double>(2, 0));
  for (int i=0;i < sample_points.size();i++) {
    sample_points[i][0] = ((double) rand()) / RAND_MAX;
    sample_points[i][1] = ((double) rand()) / RAND_MAX;
  }

  // Set up initial residual, i.e. the target function. The function we try to fit is f(x,y) = sin(2pi (x+y))^2 sin(2pi (x-y^2))
  vector<double> residual(sample_point_count, 0);
  vector<double> target_function(sample_point_count, 0);
  for (int i=0; i < residual.size(); i++) {
    double x = sample_points[i][0];
    double y = sample_points[i][1];
    residual[i] = pow(sin(M_PI*(x+y)), 2) * sin(M_PI*(x - y*y));
    target_function[i] =  pow(sin(M_PI*(x+y)), 2) * sin(M_PI*(x - y*y));
  }
  
  // Set up files for output.
  ofstream res_norm_out;
  ofstream function_one;
  ofstream function_two;
  ofstream function_three;
  ofstream original_function;
  res_norm_out.open("res_norm.csv");
  function_one.open("function_one.csv");
  function_two.open("function_two.csv");
  function_three.open("function_three.csv");

  // Plot target function.
  original_function.open("original_function.csv");
  for (int i = 0; i < target_function.size(); i++) original_function << sample_points[i][0] << ", " << sample_points[i][1] << ", " << target_function[i] << "\n";

  vector<vector<double> > prev_basis;
  for (int it = 0; it < 101; it++) {
    // Calculate and output residual norm.
    double sq_norm = 0;
    for (int j=0; j < residual.size(); j++) sq_norm += residual[j] * residual[j];
    std::cout << "Iteration: " << it << " residual norm: " << sqrt(sq_norm) << "\n";
    if (it > 0) res_norm_out << it << ", " << sqrt(sq_norm) << "\n";

    // Find dictionary element which maximizes inner product.
    double w_1, w_2, b;
    find_optimal_hyperplane(sample_points, residual, w_1, w_2, b);
    vector<double> dict_elem(sample_points.size(), 0);
    double inner_prod = 0;
    for (int i = 0; i < dict_elem.size(); i++) {
      if (w_1 * sample_points[i][0] + w_2 * sample_points[i][1] + b >= 0) {
        dict_elem[i] = 1.0;
        inner_prod += residual[i];
      }
    }
    
    // Project dictionary element orthogonal to previous elements.
    for (int i = 0; i < prev_basis.size(); i++) {
      double inner_prod = 0;
      for (int j = 0; j < dict_elem.size(); j++) {
        inner_prod += dict_elem[j] * prev_basis[i][j];
      }
      for (int j = 0; j < dict_elem.size(); j++) dict_elem[j] -= inner_prod * prev_basis[i][j];
    }

    // Normalize the new basis element.
    sq_norm = 0;
    for (int j = 0; j < dict_elem.size(); j++) sq_norm += dict_elem[j] * dict_elem[j];
    for (int j = 0; j < dict_elem.size(); j++) dict_elem[j] /= sqrt(sq_norm);

    // Obtain new residual.
    inner_prod = 0;
    for (int j = 0; j < dict_elem.size(); j++) inner_prod += dict_elem[j] * residual[j];
    for (int j = 0; j < dict_elem.size(); j++) residual[j] -= inner_prod * dict_elem[j];

    // Add basis element to list.
    prev_basis.push_back(dict_elem);
    
    // Print out the function.
    if (it == 5 || it == 10 || it == 100) {
      for (int i = 0; i < residual.size(); i++) {
        if (it == 5) function_one << sample_points[i][0] << ", " << sample_points[i][1] << ", " << target_function[i] - residual[i] << "\n";
        if (it == 10) function_two << sample_points[i][0] << ", " << sample_points[i][1] << ", " << target_function[i] - residual[i] << "\n";
        if (it == 100) function_three << sample_points[i][0] << ", " << sample_points[i][1] << ", " << target_function[i] - residual[i] << "\n";
      }
    }
  }
  res_norm_out.close();
  function_one.close();
  function_two.close();
  function_three.close();
  original_function.close();
}
