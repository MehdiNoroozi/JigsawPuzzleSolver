#! /bin/sh
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

# Test the += diagnostics.

. test-init.sh

cat >>configure.ac << 'END'
AM_CONDITIONAL([COND1], [true])
AM_CONDITIONAL([COND2], [true])
AM_CONDITIONAL([COND3], [true])
END

cat > Makefile.am << 'END'
if COND1
  C = c
if COND2
    A = a
    B = aa
    C += cc
else
    A = b
    B = bb
endif
  A += c
else
  A = d
endif
A += e

if COND3
  A += f
  B = cc
endif
B += dd
END

$ACLOCAL
AUTOMAKE_fails

# We expect the following diagnostic:
#
# Makefile.am:19: cannot apply '+=' because 'B' is not defined in
# Makefile.am:19: the following conditions:
# Makefile.am:19:   !COND1 and !COND3
# Makefile.am:19: either define 'B' in these conditions, or use
# Makefile.am:19: '+=' in the same conditions as the definitions.
#
# It would be nice if Automake could print only COND3_FALSE and
# COND1_FALSE (merging the last two conditions), so we'll support
# this case in the check too.

grep '[cC]annot apply.*+=' stderr
grep ':   !COND1 and !COND3$' stderr
# Make sure there is exactly one missing condition.
test $(grep -c ':  ' stderr) -eq 1

:
