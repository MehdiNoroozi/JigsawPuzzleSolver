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

# Make sure Makefile.in are up to date after make dist.
# This is expected to work even without GNU Make (the GNU Make
# feature that isn't supported elsewhere is the rebuild of
# Makefile dependencies during ordinary builds).
#
# If this fails, this is likely to be due to a dependency being
# given two different name.  For instance BSD Make does not know
# that 'Makefile' is the same as './Makefile'
#
# Report from Akim Demaille.

. test-init.sh

cat >>configure.ac <<'EOF'
# Rebuild rule are ok until make dist, but not afterwards.
if test ! -f rebuild_ok; then
  ACLOCAL=false
  AUTOMAKE=false
  AUTOCONF=false
fi
AC_OUTPUT
EOF

: > rebuild_ok
: > Makefile.am

$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing
./configure
$MAKE
$sleep
touch aclocal.m4
$MAKE distdir
cd $me-1.0
test ! -f rebuild_ok
./configure
$MAKE

:
