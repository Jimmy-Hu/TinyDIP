/* Developed by Jimmy Hu */
/* Cross-Platform Dynamic Shared Object (DLL/SO) Loader */

#ifndef TINYDIP_DYNAMIC_LOADER_H
#define TINYDIP_DYNAMIC_LOADER_H

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

class DynamicLibrary
{
public:
    DynamicLibrary(const std::filesystem::path& library_path)
    {
        const std::string path_str = library_path.string();
        
#ifdef _WIN32
        handle_ = LoadLibraryA(path_str.c_str());
        if (!handle_)
        {
            throw std::runtime_error("Failed to load DLL: " + path_str);
        }
#else
        // RTLD_NOW ensures all symbols are resolved, RTLD_GLOBAL is crucial 
        // for cross-boundary std::any typeid matching!
        handle_ = dlopen(path_str.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!handle_)
        {
            const char* err = dlerror();
            throw std::runtime_error("Failed to load Shared Object: " + path_str + "\nReason: " + (err ? err : "Unknown"));
        }
#endif
    }

    ~DynamicLibrary()
    {
        if (handle_)
        {
#ifdef _WIN32
            FreeLibrary(static_cast<HMODULE>(handle_));
#else
            dlclose(handle_);
#endif
        }
    }

    // Disable copy to maintain strict resource ownership (RAII)
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    // Enable move semantics
    DynamicLibrary(DynamicLibrary&& other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept
    {
        if (this != &other)
        {
            if (handle_)
            {
#ifdef _WIN32
                FreeLibrary(static_cast<HMODULE>(handle_));
#else
                dlclose(handle_);
#endif
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    template <typename Signature>
    Signature get_function(const std::string_view symbol_name) const
    {
#ifdef _WIN32
        void* symbol = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), std::string(symbol_name).c_str()));
#else
        void* symbol = dlsym(handle_, std::string(symbol_name).c_str());
#endif
        if (!symbol)
        {
            throw std::runtime_error(std::string("Failed to locate symbol: ") + std::string(symbol_name));
        }
        
        // reinterpret_cast is mandatory when bridging C-ABI boundaries into C++ typed function pointers
        return reinterpret_cast<Signature>(symbol);
    }

private:
    void* handle_{nullptr};
};

#endif