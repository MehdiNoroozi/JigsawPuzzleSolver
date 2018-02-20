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

# Another test for -Wportability warning about %-style rules, plus
# make sure we don't warn about duplicate definition for
# '${ARCH}/%.$(OBJEXT):'.
# Report from Ralf Corsepius.

. test-init.sh

cat >>Makefile.am << 'EOF'
${ARCH}/%.$(OBJEXT): %.S
	test -d ${ARCH} || mkdir ${ARCH}
	${CCASCOMPILE} -o $@ -c $<

${ARCH}/%.$(OBJEXT): %.c
	test -d ${ARCH} || mkdir ${ARCH}
	${COMPILE} -o $@ -c $<
EOF

$ACLOCAL
AUTOMAKE_fails
grep '%.*pattern.*rules' stderr

# No error otherwise.
$AUTOMAKE -Wno-portability
