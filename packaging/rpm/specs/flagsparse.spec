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
    --root %{buildroot} dist/*.whl
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
# Globbed, not %%{version}: the dist-info directory is named for the version in
# pyproject.toml, which is not necessarily this spec's Version -- a release
# pipeline may stamp the spec to the version the release manifest names while
# the Python metadata keeps its own. Binding the two made the build fail with
# "Directory not found" rather than produce a package with a mismatched
# version, which is the better failure but still a failure.
%{python3_sitelib}/flagsparse-*.dist-info/
%endif

%changelog
* Sat Jul 11 2026 FlagOS Contributors <contact@flagos.io> - 1.0.0-1
- Add pip-based fallback for distros without pyproject-rpm-macros
  (openEuler 24.03); Fedora build path unchanged.

* Wed May 13 2026 FlagOS Contributors <contact@flagos.io> - 1.0.0-1
- Initial RPM packaging.
