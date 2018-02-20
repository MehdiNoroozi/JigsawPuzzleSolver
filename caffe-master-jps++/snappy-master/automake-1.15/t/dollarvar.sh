#!/bin/sh
# Copyright (C) 2009-2014 Free Software Foundation, Inc.
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

# Test to make sure that -Wportability complains about recursive
# variable expansions and variables containing '$', '$(...)', or
# '${...}' in the name.  We support recursive variable expansions using
# the latter two constructs for the 'silent-rules' option, and they are
# rather widely supported in practice.  OTOH variable definitions
# containing a '$' on the left hand side of an assignment are not
# portable in practice, even though POSIX allows them.  :-/

. test-init.sh

cat >Makefile.am <<'EOF'
x = 1
foo$x = 1
bar$(x) = 1
baz${x} = 1
bla = $(foo$x)
bli = $(foo$(x))
blo = $(foo${x})
EOF

$ACLOCAL

AUTOMAKE_fails -Wportability
grep 'Makefile.am:2' stderr
grep 'Makefile.am:3' stderr
grep 'Makefile.am:4' stderr
grep 'Makefile.am:5' stderr
grep 'Makefile.am:6' stderr
grep 'Makefile.am:7' stderr

AUTOMAKE_fails -Wportability -Wno-portability-recursive
grep 'Makefile.am:2' stderr
grep 'Makefile.am:3' stderr
grep 'Makefile.am:4' stderr
grep 'Makefile.am:5' stderr
grep 'Makefile.am:6' stderr && exit 1
grep 'Makefile.am:7' stderr && exit 1


:
