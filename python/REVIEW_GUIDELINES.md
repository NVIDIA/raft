# AI Code Review Guidelines - RAFT Python

**Role**: Act as a principal engineer with 10+ years experience in Python systems programming and GPU memory management. Focus ONLY on CRITICAL and HIGH issues.

**Target**: Sub-3% false positive rate. Be direct, concise, minimal.

**Context**: The RAFT Python layer (pylibraft and raft-dask) provides pythonic interfaces and handle/resource management for RAFT primitives on the GPU. For formatting and pre-commit setup, see the pre-commit section of `docs/source/contributing.md`.

## IGNORE These Issues

- Style/formatting (pre-commit hooks handle this)
- Minor naming preferences (unless truly misleading)
- Personal taste on implementation (unless impacts maintainability)
- Nits that don't affect functionality
- Already-covered issues (one comment per root cause)

## CRITICAL Issues (Always Comment)

### Memory Safety
- Memory leaks from improper resource management
- Use-after-free scenarios in device memory handling
- Incorrect lifetime management of memory resources
- **Cython memory management errors** (missing `del`, incorrect reference counting)

### API Breaking Changes
- Python API changes breaking backward compatibility
- Changes to public interfaces
- Removing or renaming public methods/attributes without deprecation
- We usually require at least one release cycle for deprecations

### Integration Errors
- Incorrect handling of CuPy/Numba/PyTorch array interfaces
- Silent data corruption from type coercion
- Missing validation causing crashes on invalid input
- **Incorrect CUDA stream handling in Python bindings**

### Resource Management
- GPU memory leaks from Python objects
- Missing cleanup in `__del__` or context managers
- Circular references preventing garbage collection
- **Incorrect ownership semantics** between Python and C layer

## HIGH Issues (Comment if Substantial)

### Memory Resource Management
- Incorrect memory resource lifecycle
- Missing validation of memory resource parameters
- Improper upstream resource handling in Python

### Input Validation
- Missing size/type checks
- Not handling edge cases

### Test Quality
- Missing edge case coverage (zero-size, alignment)
- **Using external datasets** (tests must not depend on external resources)
- Missing tests for different array types (CuPy, Numba)
- **Using test classes instead of standalone functions** (RAFT prefers `test_foo_bar()` functions over `class TestFoo`)

### Documentation
- Missing or incorrect docstrings for public methods
- Parameters not documented
- Missing usage examples for memory resources
- **New public API not added to docs**

## MEDIUM Issues (Comment Selectively)

- Edge cases not handled
- Missing input validation for edge cases
- Deprecated API usage
- Minor inefficiencies in non-critical code paths

## Review Protocol

1. **Memory safety**: Resource cleanup correct? Lifetime management?
2. **API stability**: Breaking changes to Python APIs?
3. **Integration**: CuPy/Numba compatibility maintained?
4. **Input validation**: Size/type checks present?
5. **Documentation**: Public API documented?
6. **Ask, don't tell**: "Have you considered X?" not "You should do X"

## Quality Threshold

Before commenting, ask:
1. Is this actually wrong/risky, or just different?
2. Would this cause a real problem (crash, leak, API break)?
3. Does this comment add unique value?

**If no to any: Skip the comment.**

## Output Format

- Use severity labels: CRITICAL, HIGH, MEDIUM
- Be concise: One-line issue summary + one-line impact
- Provide code suggestions when you have concrete fixes
- No preamble or sign-off

## Python-Specific Considerations

**Memory Resource Lifecycle**:
- Memory resources should use context managers where appropriate
- Ensure proper cleanup in `__del__` methods
- Document ownership semantics clearly

**Cython Bindings**:
- Use proper memory management (`__dealloc__`)
- Handle exceptions correctly across Python/C++ boundary
- Ensure GIL handling is correct for CUDA operations

**Array Interfaces**:
- Support `__cuda_array_interface__` for interoperability with CuPy and pytorch
- Handle different array types (CuPy, Numba DeviceNDArray)
- Preserve array attributes where appropriate

**Error Messages**:
- Error messages must be clear and actionable for users
- Include expected vs actual values where helpful

**Testing**:
- Test different array types
- Use standalone `test_foo_bar()` functions, not test classes
- Use synthetic data, never external resources

---

## Common Bug Patterns

### 1. Resource Cleanup Issues
**Pattern**: GPU memory not properly released

**Red flags**:
- Missing `__del__` or `__dealloc__` methods
- Cleanup not happening on exception paths
- Circular references preventing garbage collection

### 2. Array Interface Errors
**Pattern**: Incorrect `__cuda_array_interface__` implementation

**Red flags**:
- Wrong shape/strides in interface dict
- Missing required keys
- Incorrect data pointer

### 3. Lifetime Management
**Pattern**: Python object outliving underlying C++ resource

**Red flags**:
- Weak references to memory resources
- Callbacks holding stale references
- Missing ref-counting in Cython

### 4. Stream Handling
**Pattern**: Incorrect stream semantics in Python bindings

**Red flags**:
- Missing stream parameter propagation
- Incorrect stream synchronization
- Default stream used when explicit stream expected

---

## Code Review Checklists

### When Reviewing Memory Resource Classes
- [ ] Is cleanup implemented in `__del__` or context manager?
- [ ] Is ownership clearly documented?
- [ ] Are edge cases handled (zero-size)?
- [ ] Is the API consistent with existing RAFT patterns?

### When Reviewing Cython Code
- [ ] Is `__dealloc__` implemented correctly?
- [ ] Are exceptions handled across the Python/C++ boundary?
- [ ] Is GIL handling correct?
- [ ] Is memory management correct (no leaks, no double-free)?
- [ ] Are errors from C/C++ calls checked and surfaced as Python exceptions?

### When Reviewing Tests
- [ ] Are different array types tested?
- [ ] Are tests written as standalone functions?

---

**Remember**: Focus on correctness and API compatibility. Catch real bugs (leaks, crashes, API
breaks), ignore style preferences. For RAFT Python: memory safety and proper resource lifecycle are paramount.
