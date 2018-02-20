#!/bin/sh
# Copyright (C) 2002-2014 Free Software Foundation, Inc.
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

# Check that conditional primaries can use directories defined
# in the same conditions (but not others).

. test-init.sh

cat >>configure.ac << 'EOF'
AM_CONDITIONAL([USE_FOO], [true])
EOF

cat >Makefile.am << 'EOF'
if USE_FOO
mydir = /foo
endif
if USE_FOO
my_DATA = foo
endif
EOF

$ACLOCAL
$AUTOMAKE

:
