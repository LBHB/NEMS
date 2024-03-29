"""A collection of tools for representing sound waveforms with spectrograms.

The contents of this package were derived from the gammatone toolkit, which is
licensed under the 3-clause BSD license. The full copyright disclaimer can be
found below, or at the following URL:
https://github.com/detly/gammatone/blob/master/COPYING

Modifications consist of standardizing docummentation format, removing outdated
documentation, renaming variables or refactoring syntax, and removing unused
code. The underlying algorithms have not been substantively changed.

Copyright
---------
Copyright (c) 1998, Malcolm Slaney <malcolm@interval.com>
Copyright (c) 2009, Dan Ellis <dpwe@ee.columbia.edu>
Copyright (c) 2014, Jason Heeris <jason.heeris@gmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from .fft import spectrogram, fft_gammagram  # Faster
from .gammatone import gammagram             # More biologically accurate
