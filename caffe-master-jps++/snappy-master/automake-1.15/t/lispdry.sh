#! /bin/sh
# Copyright (C) 2005-2014 Free Software Foundation, Inc.
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

# Check that 'make -n' works with the lisp_LISP recover rule.

required='emacs non-root'
. test-init.sh

cat > Makefile.am << 'EOF'
dist_lisp_LISP = am-one.el am-two.el am-three.el
EOF

cat >> configure.ac << 'EOF'
AM_PATH_LISPDIR
AC_OUTPUT
EOF

echo "(require 'am-two)" > am-one.el
echo "(require 'am-three) (provide 'am-two)" > am-two.el
echo "(provide 'am-three)" > am-three.el

$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing
./configure

$MAKE

test -f am-one.elc
test -f am-two.elc
test -f am-three.elc

rm -f am-*.elc

chmod a-w .

$MAKE -n

test ! -e am-one.elc
test ! -e am-two.elc
test ! -e am-three.elc

:
