# DoubleMatrix

A simple implementation for a double matrix library.  
Features;
- Hides different formats for user.
- Supports SPARSE (COO), DENSE and HASHTABLE and VECTORS.

To set globally the default format:
```c
set_default_matrix_format(HASHTABLE); // DENSE, SPARSE, HASHTABLE or VECTOR
```

| Types              | Constructors                                                 | Getter/Setter                                                | IO                                                           | Math                                                         |
|--------------------|--------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|
| Standard (ALL)<br> | dm_create<br>dm_clone<br>dm_destroy                          | dm_set<br>dm_get<br>dm_resize                                | dm_print<br>dm_brief<br>                                     | dm_transpose<br>dm_equal_matrix<br>                          |
| Special (ALL)      | dm_create_identity<br>dm_create_rand<br>dm_create_from_array<br>dm_create_diagonal | dm_push_column<br>dm_push_row<br>dm_get_sub_matrix<br><br><br> | dm_read_matrix_market<br>dm_write_matrix_market<br>dm_print_structure<br>dm_print_braille | dm_multiply_by_matrix<br>dm_multiply_by_scalar<br>dm_multiply_by_vector<br>dm_determinant<br>dm_inverse<br>dm_trace<br>dm_density<br>dm_rank<br> |
| SPARSE             | dm_create_nnz                                                |                                                              | dm_print_condensed<br>dm_brief_sparse                        |                                                              |
| VECTOR             | dv_create<br>dv_clone<br>dv_destroy                          |                                                              | dv_print<br>                                                 | dv_equal<br>dv_mean<br>dv_magnitude<br>dv_normalize(<br>dv_dot_product<br>dv_add_vector<br>dv_sub_vector<br>dv_multiply_by_scalar<br> |



   
