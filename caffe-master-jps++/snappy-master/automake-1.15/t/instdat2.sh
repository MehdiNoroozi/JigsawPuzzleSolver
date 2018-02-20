#! /bin/sh
# Copyright (C) 2001-2014 Free Software Foundation, Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Test that installing under $exec_prefix is handled by install-exec.
# Testing with headers for instance.

. test-init.sh

cat >Makefile.am << 'EOF'
# User directories.
inclexecdir = $(exec_prefix)/include
inclexec_HEADERS = my-config.h

incldatadir = $(prefix)/include
incldata_HEADERS = my-data.h

## Standard directories: _DATA
## Commented out are invalid combinations.
##bin_DATA = data
##sbin_DATA = data
##libexec_DATA = data
data_DATA = data
sysconf_DATA = data
localstate_DATA = data
##lib_DATA = data
##info_DATA = data
##man_DATA = data
##include_DATA = data
##oldinclude_DATA = data
pkgdata_DATA = data
##pkglib_DATA = data
##pkginclude_DATA = data

## Standard directories: _SCRIPTS
## Commented out are invalid combinations.
bin_SCRIPTS = script
sbin_SCRIPTS = script
libexec_SCRIPTS = script
##data_SCRIPTS = script
##sysconf_SCRIPTS = script
##localstate_SCRIPTS = script
##lib_SCRIPTS = script
##info_SCRIPTS = script
##man_SCRIPTS = script
##include_SCRIPTS = script
##oldinclude_SCRIPTS = script
pkgdata_SCRIPTS = script
##pkglib_SCRIPTS = script
##pkginclude_SCRIPTS = script
EOF

$ACLOCAL
$AUTOMAKE

# install-SCRIPTS targets.
sed -n '/^install-data-am/,/^	/p' Makefile.in > produced

cat > expected <<'EOF'
install-data-am: install-dataDATA install-incldataHEADERS \
	install-pkgdataDATA install-pkgdataSCRIPTS
EOF

diff expected produced

# install-exec targets.
sed -n '/^install-exec-am/,/^	/p' Makefile.in > produced

cat > expected <<'EOF'
install-exec-am: install-binSCRIPTS install-inclexecHEADERS \
	install-libexecSCRIPTS install-localstateDATA \
EOF

diff expected produced

:
