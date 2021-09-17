// Copyright 2019-2021 Cambridge Quantum Computing
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _TKET_BINDER_UTILS_H_
#define _TKET_BINDER_UTILS_H_

/** Narrow No-Break Space (U+202F, UTF-8 encoding) */
#define NNBSP "\xE2\x80\xAF"

/** Pluralize a reference to a class object in a docstring */
#define CLSOBJS(a) ":py:class:`" #a "`" NNBSP "s"

#endif