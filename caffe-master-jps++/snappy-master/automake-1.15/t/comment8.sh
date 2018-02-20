#! /bin/sh
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

# Make sure += does not append to a comment.
# Report from Stepan Kasal.

. test-init.sh

cat >> configure.ac <<'EOF'
AM_CONDITIONAL([COND1], [true])
AM_CONDITIONAL([COND2], [true])
AC_OUTPUT
EOF

cat > Makefile.am << 'EOF'
VAR = valA# comA ## com C
VAR += valB # comB
if COND1
  VAR += val1 # com1
endif COND1
VAR += valC
if COND2
  VAR += val2 # com2
endif COND2

.PHONY: test
test:
	is $(VAR) == valA valB val1 valC val2
EOF

$ACLOCAL
$AUTOCONF
$AUTOMAKE
./configure
$MAKE test
