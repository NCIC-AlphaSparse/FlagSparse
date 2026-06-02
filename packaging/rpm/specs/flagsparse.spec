%global debug_package %{nil}

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
BuildRequires:  pyproject-rpm-macros

%description
Sparse matrix operators (SpMM, SpMV, sampled dense-dense) for FlagOS-supported accelerators.

%prep
%autosetup -n flagsparse-%{version}

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files flagsparse

%check
# Smoke find_spec test (no actual import) — verifies the built module
# lands at the expected sitelib path. Doesn't import the module so
# missing runtime deps (torch, triton, ...) don't trip the check;
# those are user-install-time concerns, not packaging concerns.
PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=%{buildroot}%{python3_sitelib} \
    python3 -c "import importlib.util; s = importlib.util.find_spec('flagsparse'); assert s and s.origin, 'flagsparse not findable'; print('OK: flagsparse at', s.origin)"

%files -f %{pyproject_files}
%license LICENSE

%changelog
* Wed May 13 2026 FlagOS Contributors <contact@flagos.io> - 1.0.0-1
- Initial RPM packaging.
