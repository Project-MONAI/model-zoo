import tempfile
import cupy as cp
import kvikio

def warm_up():
    a = cp.arange(100)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file_name = tmp_file.name
        f = kvikio.CuFile(tmp_file_name, "w")
        # Write whole array to file
        f.write(a)
        f.close()

        b = cp.empty_like(a)
        f = kvikio.CuFile(tmp_file_name, "r")
        f.read(b)
