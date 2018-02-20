#! /bin/sh
# Copyright (C) 2003-2014 Free Software Foundation, Inc.
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

# Automake should not assume that make files are called Makefile.
# Report from Braden McDaniel.

required=GNUmake
. test-init.sh

cat >> configure.ac << 'END'
AC_CONFIG_FILES([sub/GNUmakefile])
AC_OUTPUT
END

mkdir sub

echo SUBDIRS = sub > Makefile.am

cat > sub/GNUmakefile.am <<'EOF'
# In this project, the Makefile is an installed data file.
dist_data_DATA = Makefile
EOF

echo 'this should not cause any problem' > sub/Makefile

$ACLOCAL
$AUTOCONF
$AUTOMAKE
./configure
$MAKE distcheck
