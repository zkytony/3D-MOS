## 1. Installation <a name="installation"/>

The required python version is Python 3.6+.

1. Clone the repository and create and virtual environment with the following lines.

    ```
    git clone git@github.com:zkytony/3D-MOS.git
    cd 3D-MOS;
    virtualenv -p $(which python3) venv/mos3d
    source venv/mos3d/bin/activate
    ```

2. Install [pomdp-py](https://github.com/h2r/pomdp-py)

    ```
    pip install pomdp-py
    ```

3. Install the `mos3d` package. Assume you're at the root of the repository.

    ```
    pip install -e .
    ```

4. Test

   ```
   cd tests/
   python test_models.py
   python test_octree_belief.py
   python test_sensor.py
   python test_abstraction.py
   ```

   ![output_test_abstraction.png](docs/figs/sim-example-occ.png)
