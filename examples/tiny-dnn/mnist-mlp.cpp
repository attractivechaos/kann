/*
   Copyright (c) 2013, Taiga Nomi
   All rights reserved.

   Use of this source code is governed by a BSD-style license that can be found
   in the LICENSE file.
 */
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <stdio.h>

#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

using namespace std;

network<sequential> mlp_model_gen(int n_in, int n_out, int n_layer, int n_hidden)
{
	network<sequential> nn;
	int n_last = n_in;
	for (int i = 0; i < n_layer; ++i) {
		nn << fully_connected_layer<relu>(n_last, n_hidden);
		n_last = n_hidden;
	}
	nn << fully_connected_layer<softmax>(n_last, n_out);
	return nn;
}

void sample2_mlp(const string& data_dir)
{
	const serial_size_t n_hidden = 64, mini_size = 64;
	auto nn = mlp_model_gen(28 * 28, 10, 1, n_hidden);
	
	gradient_descent optimizer;

	// load MNIST dataset
	std::vector<label_t> train_labels, test_labels;
	std::vector<vec_t> train_images, test_images;

	std::string train_labels_path = data_dir + "/train-labels-idx1-ubyte";
	std::string train_images_path = data_dir + "/train-images-idx3-ubyte";
	std::string test_labels_path  = data_dir + "/t10k-labels-idx1-ubyte";
	std::string test_images_path  = data_dir + "/t10k-images-idx3-ubyte";

	parse_mnist_labels(train_labels_path, &train_labels);
	parse_mnist_images(train_images_path, &train_images, -1.0, 1.0, 0, 0);
	parse_mnist_labels(test_labels_path, &test_labels);
	parse_mnist_images(test_images_path, &test_images, -1.0, 1.0, 0, 0);

	optimizer.alpha = 0.001 * mini_size;

	progress_display disp(train_images.size());
	timer t;

	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << t.elapsed() << "s elapsed." << std::endl;

		tiny_dnn::result res = nn.test(test_images, test_labels);

		std::cout << optimizer.alpha << "," << res.num_success << "/"
			<< res.num_total << std::endl;

		optimizer.alpha *= 0.85;  // decay learning rate
		optimizer.alpha = std::max((tiny_dnn::float_t)0.00001 * mini_size, optimizer.alpha);

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_data = [&]() { disp += mini_size; };

	nn.train<cross_entropy_multiclass>(optimizer, train_images, train_labels, mini_size, 20, on_enumerate_data, on_enumerate_epoch);
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		cerr << "Usage: mnist-mlp <data-dir>" << std::endl;
		return 1;
	}
	sample2_mlp(argv[1]);
	return 0;
}
