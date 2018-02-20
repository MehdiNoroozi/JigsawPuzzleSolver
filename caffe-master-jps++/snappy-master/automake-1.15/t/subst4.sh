#! /bin/sh
# Copyright (C) 2008-2014 Free Software Foundation, Inc.
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

# Test AM_SUBST_NOTMAKE.

. test-init.sh

cat >> configure.ac <<'EOF'
myrule="\
foo: bar
	echo making \$@ from bar
	echo \$@ > \$@
"
AC_SUBST([myrule])
AM_SUBST_NOTMAKE([myrule])
AC_OUTPUT
EOF

cat > Makefile.am <<'EOF'
@myrule@
EOF
: > bar

$ACLOCAL
$AUTOCONF
$AUTOMAKE
./configure
$MAKE foo
test -f foo

:
