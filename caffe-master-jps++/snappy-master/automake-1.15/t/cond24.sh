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

# Check that conditional redefinitions of AC_SUBST'ed variables are detected.
# Report from Patrik Weiskircher.

. test-init.sh

cat >>configure.ac <<EOF
AC_SUBST([foo], [bar])
AM_CONDITIONAL([COND], [true])
EOF

cat >Makefile.am <<EOF
if COND
## A dummy comment to change line numer.
foo = baz
endif
EOF

$ACLOCAL
AUTOMAKE_fails
grep '^Makefile\.am:3:.* foo was already defined' stderr
grep '^configure\.ac:4:.*foo.* previously defined here' stderr

:
