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

# Test parallel automake execution.

# This tests:
# 4) warning and normal error output should be identical, in that duplicate
#    warnings should be omitted in the same way as without threads.

required=perl-threads
. test-init.sh

mkdir sub

cat > Makefile.am << 'END'
AUTOMAKE_OPTIONS = subdir-objects
bin_PROGRAMS = main
main_SOURCES = sub/main.c
SUBDIRS =
END

list='1 2 3'
for i in $list; do
  echo "AC_CONFIG_FILES([sub$i/Makefile])" >> configure.ac
  echo "SUBDIRS += sub$i" >> Makefile.am
  mkdir sub$i sub$i/sub
  unindent > sub$i/Makefile.am << END
    AUTOMAKE_OPTIONS = subdir-objects
    bin_PROGRAMS = sub$i
    sub${i}_SOURCES = sub/main$i.c
END
done

mkdir build-aux

$ACLOCAL

# Independently of the number of worker threads, automake output
# should be
# - stable (multiple runs should produce the same output),
# - properly uniquified,
# - complete (output from worker threads should not be lost).
#
# The parts output by --add-missing are unstable not only wrt. order
# but also wrt. content: any of the Makefile.am files may cause the
# depcomp script to be installed (or several of them).
# Thus we install the auxiliary files in a prior step.

# Generate expected output using non-threaded code.
unset AUTOMAKE_JOBS
rm -f install-sh missing depcomp
AUTOMAKE_fails --add-missing
mv stderr expected

AUTOMAKE_JOBS=5
export AUTOMAKE_JOBS

for i in 1 2 3 4 5 6 7 8; do
  rm -f install-sh missing depcomp
  AUTOMAKE_fails --add-missing
  diff expected stderr
done

:
