# mnist-cnn -> kad_vleaf

- kad_vleaf

# mnist-cnn -> kad_relu -> kann_layer_conv2d

- kad_trap_fe

- kann_srand

- kann_load

  - kann_load_fp
    - kad_size_const
    - kad_size_var
    - kad_load
    - kad_ext_sync

- kad_feed

  - kad_vleaf

- kad_relu

  - KAD_FUNC_OP1
    - kad_op1_core

- kann_layer_conv2d

  - kann_new_weight_conv2d
    - kann_new_leaf
    - kann_new_leaf_array
    - kad_len
    - kad_drand_normal
    - kad_drand
    - kad_xoroshiro128plus_next

- kad_max2d

  - kad_new_core
  - conv2d_gen_aux
    - conv_find_par
  - kad_finalize_node
    - kad_finalize_node

- kann_layer_dropout

  - kann_layer_dropout2
    - kann_new_leaf2
      - kann_new_leaf_array
        - kad_len

- kann_layer_dense

- kann_new

- kann_layer_cost

---

- kann_new_weight_conv2d

  - kann_new_leaf
  - kann_new_leaf_array
  - kad_len
  - kad_drand_normal
  - kad_drand
  - kad_xoroshiro128plus_next

- kad_conv2d
- kad_new_core
- conv2d_gen_aux
  - conv_find_par
- kad_finalize_node

mnist-cnn
mnist-cnn
mnist-cnn
mnist-cnn
mnist-cnn
mnist-cnn
mnist-cnn
mnist-cnn
mnist-cnn
