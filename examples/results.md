# REGULAR

test 1:
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 228 sec and 731067146 nsec (228.731067146 sec)

test 2:
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 234 sec and 543783792 nsec (234.543783792 sec)

C took 230 sec and 392200663 nsec (230.392200663 sec)

# OPT

epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 220 sec and 831098514 nsec (220.831098514 sec)

[sharithg@scc-204 examples]$ ./mnist-cnn_opt -o mnist-cnn.kan -t4 kann-data/mnist-train-?.knd.gz
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 218 sec and 521811693 nsec (218.521811693 sec)

epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 218 sec and 599231380 nsec (218.599231380 sec)

# SSE 256 (kad_saxpy_inlined)

epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 187 sec and 464088324 nsec (187.464088324 sec)

# SSE 256 (kad_sdot)

epoch: 1; training cost: 0.0522196 (class error: 8.09%); validation cost: 0.0153723 (class error: 2.10%)
C took 181 sec and 838404501 nsec (181.838404501 sec)

# Training 10 (Naive)

--------------Running test 1----------------------------
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 248 sec and 515398818 nsec (248.515398818 sec)
--------------Running test 2----------------------------
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 250 sec and 709382150 nsec (250.709382150 sec)
--------------Running test 3----------------------------
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 253 sec and 37881755 nsec (253.037881755 sec)
--------------Running test 4----------------------------
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 253 sec and 151411236 nsec (253.151411236 sec)
--------------Running test 5----------------------------
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 251 sec and 136416851 nsec (251.136416851 sec)
--------------Running test 6----------------------------
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 249 sec and 702907316 nsec (249.702907316 sec)
--------------Running test 7----------------------------
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 251 sec and 241454354 nsec (251.241454354 sec)
--------------Running test 8----------------------------
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 253 sec and 100908962 nsec (253.100908962 sec)
--------------Running test 9----------------------------
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 253 sec and 695699643 nsec (253.695699643 sec)
--------------Running test 10----------------------------
epoch: 1; training cost: 0.0521867 (class error: 8.13%); validation cost: 0.015372 (class error: 2.27%)
C took 251 sec and 901630912 nsec (251.901630912 sec)
