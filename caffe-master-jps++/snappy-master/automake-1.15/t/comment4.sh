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

# Make sure commented variables are output near their comments.

. test-init.sh

cat >> configure.ac <<'EOF'
AC_OUTPUT
EOF

cat > Makefile.am << 'EOF'
# UnIqUe_COPYRIGHT_BOILERPLATE

# UnIqUe_MUMBLE_COMMENT
mumble = UnIqUe_MUMBLE_VALUE
EOF

$ACLOCAL
$AUTOMAKE
# UnIqUe_COPYRIGHT_BOILERPLATE should appear near the top of the file.
test $(sed -n -e '1,/UnIqUe_COPYRIGHT_BOILERPLATE/p' \
                 Makefile.in | wc -l) -le 30
# UnIqUe_MUMBLE_COMMENT should appear right before the mumble declaration.
test $(sed -n -e '/UnIqUe_MUMBLE_COMMENT/,/UnIqUe_MUMBLE_VALUE/p' \
                 Makefile.in | wc -l) -eq 2

:
