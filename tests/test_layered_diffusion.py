"""Analytic fixtures and separate native smoke for the layered extension."""
import pytest
from conftest import requires_mfem
from benchmarks.layered_diffusion import exact, flux, verify_field, run


def fixture(path, n, kappa, perturb=0):
    rows=[]
    for i in range(n):
        for j in range(n):
            for x,y in ((i,j),(i+1,j),(i,j+1),(i+1,j+1)):
                u=exact(x/n,kappa)
                if (x,y)==(1,1):
                    u+=perturb
                rows.append(f"{i*n+j} {x/n} {y/n} {u}")
    path.write_text("\n".join(rows)+"\n")
    return path


@pytest.mark.parametrize("kappa",[1,100,1e4,1e6])
def test_analytic_field(tmp_path,kappa):
    assert exact(0,kappa)==0
    assert exact(1,kappa)==pytest.approx(1)
    assert 1<=flux(kappa)<=2
    r=verify_field(fixture(tmp_path/"field.txt",4,kappa),4,kappa)
    assert r["accepted"]


def test_wrong_field_rejected(tmp_path):
    r=verify_field(fixture(tmp_path/"field.txt",4,100,1e-4),4,100)
    assert not r["accepted"]


def test_incomplete_mesh_rejected(tmp_path):
    f=fixture(tmp_path/"field.txt",4,100)
    f.write_text("\n".join(f.read_text().splitlines()[:-4]))
    with pytest.raises(ValueError,match="incomplete"):
        verify_field(f,4,100)


@pytest.mark.mfem
@requires_mfem
@pytest.mark.parametrize("kappa",[1,100])
def test_native_layered(tmp_path,kappa):
    r=run(tmp_path/"native",8,kappa)
    assert r["accepted"],r
