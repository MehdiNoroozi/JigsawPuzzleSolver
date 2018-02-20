#! /bin/sh

# This script helps bootstrap automake, when checked out from git.
#
# Copyright (C) 2002-2014 Free Software Foundation, Inc.
# Originally written by Pavel Roskin <proski@gnu.org> September 2002.
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

# Don't ignore failures.
set -e

# Set program basename.
me=`echo "$0" | sed 's,^.*/,,'`

# Let user choose which version of autoconf, autom4te and perl to use.
: ${AUTOCONF=autoconf}
export AUTOCONF  # might be used by aclocal and/or automake
: ${AUTOM4TE=autom4te}
export AUTOM4TE  # ditto
: ${PERL=perl}

# Variables to substitute.
VERSION=`sed -ne '/AC_INIT/s/^[^[]*\[[^[]*\[\([^]]*\)\].*$/\1/p' configure.ac`
PACKAGE=automake
datadir=.
# This should be automatically updated by the 'update-copyright'
# rule of our Makefile.
RELEASE_YEAR=2014

# Override SHELL.  This is required on DJGPP so that Perl's system()
# uses bash, not COMMAND.COM which doesn't quote arguments properly.
# It's not used otherwise.
if test -n "$DJDIR"; then
  BOOTSTRAP_SHELL=/dev/env/DJDIR/bin/bash.exe
else
  BOOTSTRAP_SHELL=/bin/sh
fi

# Read the rule for calculating APIVERSION and execute it.
apiver_cmd=`sed -ne 's/\[\[/[/g;s/\]\]/]/g;/^APIVERSION=/p' configure.ac`
eval "$apiver_cmd"

# Sanity checks.
if test -z "$VERSION"; then
  echo "$me: cannot find VERSION" >&2
  exit 1
fi

if test -z "$APIVERSION"; then
  echo "$me: cannot find APIVERSION" >&2
  exit 1
fi

# Make a dummy versioned directory for aclocal.
rm -rf aclocal-$APIVERSION
mkdir aclocal-$APIVERSION
if test -d automake-$APIVERSION; then
  find automake-$APIVERSION -exec chmod u+wx '{}' ';'
fi
rm -rf automake-$APIVERSION
# Can't use "ln -s lib automake-$APIVERSION", that would create a
# lib.exe stub under DJGPP 2.03.
mkdir automake-$APIVERSION
cp -rf lib/* automake-$APIVERSION

dosubst ()
{
  rm -f $2
  in=`echo $1 | sed 's,^.*/,,'`
  sed -e "s%@APIVERSION@%$APIVERSION%g" \
      -e "s%@PACKAGE@%$PACKAGE%g" \
      -e "s%@PERL@%$PERL%g" \
      -e "s%@SHELL@%$BOOTSTRAP_SHELL%g" \
      -e "s%@VERSION@%$VERSION%g" \
      -e "s%@datadir@%$datadir%g" \
      -e "s%@RELEASE_YEAR@%$RELEASE_YEAR%g" \
      -e "s%@configure_input@%Generated from $in; do not edit by hand.%g" \
      $1 > $2
  chmod a-w $2
}

# Create temporary replacement for lib/Automake/Config.pm.
dosubst automake-$APIVERSION/Automake/Config.in \
        automake-$APIVERSION/Automake/Config.pm

# Overwrite amversion.m4.
dosubst m4/amversion.in m4/amversion.m4

# Create temporary replacement for aclocal and automake.
for p in bin/aclocal bin/automake; do
  dosubst $p.in $p.tmp
  $PERL -w bin/gen-perl-protos $p.tmp > $p.tmp2
  mv -f $p.tmp2 $p.tmp
done

# Create required makefile snippets.
$PERL ./gen-testsuite-part > t/testsuite-part.tmp
chmod a-w t/testsuite-part.tmp
mv -f t/testsuite-part.tmp t/testsuite-part.am

# Run the autotools.  Bail out if any warning is triggered.
# Use '-I' here so that our own *.m4 files in m4/ gets included,
# not copied, in aclocal.m4.
$PERL ./bin/aclocal.tmp -Wall -Werror -I m4 \
                        --automake-acdir=m4 --system-acdir=m4/acdir
$AUTOCONF -Wall -Werror
$PERL ./bin/automake.tmp -Wall -Werror

# Remove temporary files and directories.
rm -rf aclocal-$APIVERSION automake-$APIVERSION
rm -f bin/aclocal.tmp bin/automake.tmp
