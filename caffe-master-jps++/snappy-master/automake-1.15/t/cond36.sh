#!/bin/sh
# Copyright (C) 2004-2014 Free Software Foundation, Inc.
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

# Check rules output for parser defined conditionally.

. test-init.sh

cat >>configure.ac <<'EOF'
AM_CONDITIONAL([CASE_A], [test -z "$case_B"])
AC_PROG_CC
AM_PROG_LEX
AC_PROG_YACC
AC_OUTPUT
EOF

cat > Makefile.am <<'EOF'
AM_YFLAGS               =       -d

BUILT_SOURCES           =       tparse.h

if CASE_A
bin_PROGRAMS            =       ta
ta_SOURCES              =       ta.c tparse.h tscan.l tparse.y
ta_LDADD                =       $(LEXLIB)
else
bin_PROGRAMS            =       tb
tb_SOURCES              =       tb.c tparse.h tscan.l tparse.y
tb_LDADD                =       $(LEXLIB)
tparse.h: tparce.c
	echo whatever
endif
EOF

$ACLOCAL

# Presently Automake doesn't fully support partially overriden rules
# and should complain.
AUTOMAKE_fails --add-missing
grep 'tparse\.h.*already defined' stderr
$AUTOMAKE -Wno-error

# Still and all, it should generate two rules.
$FGREP 'tparse.h' Makefile.in # For debugging.
test $($FGREP -c 'tparse.h:' Makefile.in) -eq 2
$FGREP '@CASE_A_TRUE@tparse.h:' Makefile.in
$FGREP '@CASE_A_FALSE@tparse.h:' Makefile.in

:
