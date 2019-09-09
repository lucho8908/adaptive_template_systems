#!/bin/bash

# python reproduce_teaspoon.py 10 > manifold_10.txt
# python reproduce_teaspoon.py 25 > manifold_25.txt
# python reproduce_teaspoon.py 50 > manifold_50.txt
# python reproduce_teaspoon.py 100 > manifold_100.txt
# python reproduce_teaspoon.py 200 > manifold_200.txt

# python reproduce_teaspoon_cder.py 10 > cder_2_manifold_10.txt
# python reproduce_teaspoon_cder.py 25 > cder_2_manifold_25.txt
# python reproduce_teaspoon_cder.py 50 > cder_2_manifold_50.txt
# python reproduce_teaspoon_cder.py 100 > cder_2_manifold_100.txt
# python reproduce_teaspoon_cder.py 200 > cder_2_manifold_200.txt

# python reproduce_teaspoon_gmm.py 10 > gmm_manifold_10.txt
# python reproduce_teaspoon_gmm.py 25 > gmm_manifold_25.txt
# python reproduce_teaspoon_gmm.py 50 > gmm_manifold_50.txt
# python reproduce_teaspoon_gmm.py 100 > gmm_manifold_100.txt
# python reproduce_teaspoon_gmm.py 200 > gmm_manifold_200.txt

python reproduce_teaspoon_hdbscan.py 10 > hdbscan_manifold_10.txt
python reproduce_teaspoon_hdbscan.py 25 > hdbscan_manifold_25.txt
python reproduce_teaspoon_hdbscan.py 50 > hdbscan_manifold_50.txt
python reproduce_teaspoon_hdbscan.py 100 > hdbscan_manifold_100.txt
python reproduce_teaspoon_hdbscan.py 200 > hdbscan_manifold_200.txt