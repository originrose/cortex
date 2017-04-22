#ifndef THINK_CUDA_ERRORS_H
#define THINK_CUDA_ERRORS_H


#include <cuda.h>
#include <cstdint>
#include <string>
#include <sstream>
#include <unordered_map>
#include <utility>


namespace think {
  using namespace std;


  typedef pair<int32_t, string> error_pair;

  
#define DEFINE_ERROR_PAIR(x) error_pair(x, #x)

  const error_pair g_error_names[] = {
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::cuEventQuery() and ::cuStreamQuery()).
     */
    DEFINE_ERROR_PAIR(CUDA_SUCCESS),
    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_INVALID_VALUE),

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_OUT_OF_MEMORY),

    /**
     * This indicates that the CUDA driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_NOT_INITIALIZED),

    /**
     * This indicates that the CUDA driver is in the process of shutting down.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_DEINITIALIZED),

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_PROFILER_DISABLED),

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cuProfilerStart or
     * ::cuProfilerStop without initialization.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_PROFILER_NOT_INITIALIZED),

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cuProfilerStart() when profiling is already enabled.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_PROFILER_ALREADY_STARTED),

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cuProfilerStop() when profiling is already disabled.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_PROFILER_ALREADY_STOPPED),

    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_NO_DEVICE),

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_INVALID_DEVICE),


    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid CUDA module.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_INVALID_IMAGE),

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_INVALID_CONTEXT),

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of CUDA 3.2. It is no longer an
     * error to attempt to push the active context via ::cuCtxPushCurrent().
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_CONTEXT_ALREADY_CURRENT),

    /**
     * This indicates that a map or register operation has failed.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_MAP_FAILED),

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_UNMAP_FAILED),

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_ARRAY_IS_MAPPED),

    /**
     * This indicates that the resource is already mapped.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_ALREADY_MAPPED),

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_NO_BINARY_FOR_GPU),

    /**
     * This indicates that a resource has already been acquired.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_ALREADY_ACQUIRED),

    /**
     * This indicates that a resource is not mapped.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_NOT_MAPPED),

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_NOT_MAPPED_AS_ARRAY),

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_NOT_MAPPED_AS_POINTER),

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_ECC_UNCORRECTABLE),

    /**
     * This indicates that the ::CUlimit passed to the API call is not
     * supported by the active device.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_UNSUPPORTED_LIMIT),

    /**
     * This indicates that the ::CUcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already 
     * bound to a CPU thread.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_CONTEXT_ALREADY_IN_USE),

    /**
     * This indicates that peer access is not supported across the given
     * devices.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED),

    /**
     * This indicates that a PTX JIT compilation failed.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_INVALID_PTX),

    /**
     * This indicates an error with OpenGL or DirectX context.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_INVALID_GRAPHICS_CONTEXT),

    /**
     * This indicates that an uncorrectable NVLink error was detected during the
     * execution.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_NVLINK_UNCORRECTABLE),

    /**
     * This indicates that the device kernel source is invalid.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_INVALID_SOURCE),

    /**
     * This indicates that the file specified was not found.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_FILE_NOT_FOUND),

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND),

    /**
     * This indicates that initialization of a shared object failed.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED),

    /**
     * This indicates that an OS call failed.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_OPERATING_SYSTEM),

    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_INVALID_HANDLE),

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names, and surface names.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_NOT_FOUND),

    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_NOT_READY),

    /**
     * While executing a kernel, the device encountered a
     * load or store instruction on an invalid memory address.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_ILLEGAL_ADDRESS),

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES),

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using CUDA.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_LAUNCH_TIMEOUT),

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING),
    
    /**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED),

    /**
     * This error indicates that ::cuCtxDisablePeerAccess() is 
     * trying to disable peer access which has not been enabled yet 
     * via ::cuCtxEnablePeerAccess(). 
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED),

    /**
     * This error indicates that the primary context for the specified device
     * has already been initialized.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE),

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_CONTEXT_IS_DESTROYED),

    /**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device 
     * memory allocations from this context are invalid and must be 
     * reconstructed if the program is to continue using CUDA.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_ASSERT),

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices 
     * passed to ::cuCtxEnablePeerAccess().
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_TOO_MANY_PEERS),

    /**
     * This error indicates that the memory range passed to ::cuMemHostRegister()
     * has already been registered.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED),

    /**
     * This error indicates that the pointer passed to ::cuMemHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED),

    /**
     * While executing a kernel, the device encountered a stack error.
     * This can be due to stack corruption or exceeding the stack size limit.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_HARDWARE_STACK_ERROR),

    /**
     * While executing a kernel, the device encountered an illegal instruction.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_ILLEGAL_INSTRUCTION),

    /**
     * While executing a kernel, the device encountered a load or store instruction
     * on a memory address which is not aligned.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_MISALIGNED_ADDRESS),

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_INVALID_ADDRESS_SPACE),

    /**
     * While executing a kernel, the device program counter wrapped its address space.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_INVALID_PC),

    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_LAUNCH_FAILED),


    /**
     * This error indicates that the attempted operation is not permitted.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_NOT_PERMITTED),

    /**
     * This error indicates that the attempted operation is not supported
     * on the current system or device.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_NOT_SUPPORTED),

    /**
     * This indicates that an unknown internal error has occurred.
     */
    DEFINE_ERROR_PAIR(CUDA_ERROR_UNKNOWN),
  
  };

  int32_t num_error_names = sizeof(g_error_names) / sizeof( error_pair );

  typedef unordered_map<int32_t,string> error_to_name_map;

  struct error_to_name
  {
    error_to_name_map m_map;
    error_to_name() {
      for( int32_t idx = 0; idx < num_error_names; ++idx )
	m_map.insert(g_error_names[idx]);
    }
    string operator()( int32_t error ) {
      auto name = m_map.find( error );
      if ( name != m_map.end() )
	return name->second;
      return "";
    }
  };

  error_to_name g_error_map;

  string error_name( int error)
  {
    string retval = g_error_map(error);
    if ( !retval.empty() )
      return retval;

    stringstream stream;
    stream << "Unknown error value " << error;
    return stream.str();

  }
}


#endif
