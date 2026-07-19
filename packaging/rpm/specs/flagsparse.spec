%global debug_package %{nil}

# Distros that ship pyproject-rpm-macros (Fedora, EL9+) build via the
# %%pyproject_* macro family — that path is unchanged. Distros without
# it (openEuler 24.03, EL8-family) fall back to a plain pip
# wheel/install build. Capability-detected at parse time, so the build
# container must have its python toolchain installed before rpmbuild
# runs (both Dockerfile.rpm paths do).
%if %{defined pyproject_wheel}
%global has_pyproject_macros 1
%else
%global has_pyproject_macros 0
%endif

Name:           python3-flagsparse
Version:        1.0.0
Release:        1%{?dist}
Summary:        FlagSparse — sparse compute kernels for FlagOS

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagSparse
Source0:        %{url}/archive/refs/tags/v%{version}.tar.gz#/flagsparse-%{version}.tar.gz
BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools >= 60
BuildRequires:  python3-wheel
BuildRequires:  python3-pip
%if %{has_pyproject_macros}
BuildRequires:  pyproject-rpm-macros
%endif

%description
Sparse matrix operators (SpMM, SpMV, sampled dense-dense) for FlagOS-supported accelerators.

%prep
%autosetup -n flagsparse-%{version}

%build
%if %{has_pyproject_macros}
%pyproject_wheel
%else
%{__python3} -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
%endif

%install
%if %{has_pyproject_macros}
%pyproject_install
%pyproject_save_files flagsparse
%else
%{__python3} -m pip install --no-deps --no-index --no-warn-script-location \
    --root %{buildroot} --prefix /usr dist/*.whl
%endif

%check
# Smoke find_spec test (no actual import) — verifies the built module
# lands at the expected sitelib path. Doesn't import the module so
# missing runtime deps (torch, triton, ...) don't trip the check;
# those are user-install-time concerns, not packaging concerns.
PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=%{buildroot}%{python3_sitelib} \
    python3 -c "import importlib.util; s = importlib.util.find_spec('flagsparse'); assert s and s.origin, 'flagsparse not findable'; print('OK: flagsparse at', s.origin)"

%if %{has_pyproject_macros}
%files -f %{pyproject_files}
%license LICENSE
%else
%files
%license LICENSE
%{python3_sitelib}/flagsparse/
%{python3_sitelib}/flagsparse-%{version}.dist-info/
%endif

%changelog
* Sat Jul 11 2026 FlagOS Contributors <contact@flagos.io> - 1.0.0-1
- Add pip-based fallback for distros without pyproject-rpm-macros
  (openEuler 24.03); Fedora build path unchanged.

* Wed May 13 2026 FlagOS Contributors <contact@flagos.io> - 1.0.0-1
- Initial RPM packaging.
