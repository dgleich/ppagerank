/*
 * David Gleich
 * 12 January 2007
 * Copyright, Stanford University, 2007
 */

/**
 * @file bvgraph_matrix.cc
 * Interprete a Boldi-Vigna Graph as a matrix.
 */
 
#include "bvgraph_matrix.h"
#include <util/file.h>

namespace yasmic 
{      
    void bvgraph_matrix::load_internal()
    {
        // read properties
        std::string propfilename = basename + ".properties";
        std::string graphfilename = basename + ".graph";
        std::ifstream propif(propfilename.c_str());
        if (!propif || !util::file_exists(graphfilename.c_str())) {
            // TODO report error
            return;
        }
        
        std::map<std::string, int> options;
        
        // initialize the options we want
        options["nodes"];
        options["arcs"];
        options["windowsize"];
        options["minintervallength"];
        options["maxrefcount"];
        options["zetak"];
        
        typedef std::string::size_type position;
        std::string property_line;
        while (std::getline(propif, property_line))
        {
            // check if there is a property listed
            position eq = property_line.find('=');
            if (eq == std::string::npos) { continue; }
            std::string key = property_line.substr(0,eq);
            // TODO throw error on non-trivial compressionflags
            if (options.find(key) != options.end())
            {
                // we want to save this key
                std::string value = property_line.substr(eq+1);
                options[key] = boost::lexical_cast<int>(value);
            }
        }
        
        n = options["nodes"];
        m = options["arcs"];
        _window_size = options["windowsize"];
        _min_interval_length = options["minintervallength"];
        _max_ref_count = options["maxrefcount"];
        _zeta_k = options["zetak"];
        
        // note that if compressionflags is specified, boost::lexical_cast 
        // will through an exception because it is an invalid type.
        if (n == 0 || m == 0 || _min_interval_length <= 1) {
            // TODO throw an error
        }
    }
        
       
    
    
    namespace impl 
    {
        const int bit_istream::BYTELSB[] = {
                -1, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0
        };
            
        const int bit_istream::BYTEMSB[] = {
                -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
        };
            
        int bit_istream::read_int(int len) 
        {
            int i, x = 0;
            assert (len >= 0);
            assert (len <= 32);
            

            if (len <= fill) return read_from_current( len );

            len -= fill;
            x = read_from_current( fill );
            i = len >> 3;
            while (i-- != 0) { x = x << 8 | read(); }
            read_bits += len & ~7;
            
            len &= 7;

            return ( x << len ) | read_from_current( len );
        }
        
        int bit_istream::read_unary() 
        {
            assert ( fill < 24 );
            int x;
            
            const unsigned int current_left_aligned = current << (24 - fill) & 0xFFFFFF;
            if (current_left_aligned != 0)
            {
                if ((current_left_aligned & 0xFF0000) != 0) { x = 7 - BYTEMSB[current_left_aligned >> 16]; }
                else if ((current_left_aligned & 0xFF00) != 0) { x = 15 - BYTEMSB[current_left_aligned >> 8]; }
                else { x = 23 - BYTEMSB[current_left_aligned & 0xFF]; }
                read_bits += x + 1;
                fill -= x + 1;
                return (x);
            }
            
            x = fill;
            while ( (current = read()) == 0) { x += 8; }
            x += 7 - ( fill = BYTEMSB[current] );
            read_bits += x + 1;
            return (x);
        }
        
        int bit_istream::read_gamma()
        {
            const int msb = read_unary();
            return ( ( 1 << msb ) | read_int(msb) ) - 1;
        }
        
        int bit_istream::read_zeta(const int k)
        {
            const int h = read_unary();
            const int left = 1 << h * k;
            const int m = read_int(h*k + k - 1);
            if (m < left) { return (m + left - 1); }
            else { return (m << 1) + read_bit() - 1; }
        }
        
        void bvgraph_sequential_iterator::load_successors()
        {
            const int x = curr;
            int ref, ref_index;
            int i, extra_count, block_count = 0;
            std::vector<int> block, left, len;
            
            int d = outd[x%cyclic_buffer_size]=read_outdegree();
            curr_outd = d;
            if (d == 0) { return; }
            
            // std::cout << "** Start successors " << x << " outdegree " << curr_outd << std::endl;
            
            // we read the reference only if the actual window size is larger than one 
            // (i.e., the one specified by the user is larger than 0).
            if ( window_size > 0 ) {
                ref = read_reference();
            }
            
            ref_index = (x - ref + cyclic_buffer_size) % cyclic_buffer_size;
            
            if (ref > 0)
            {
                if ( (block_count = read_block_count()) != 0 ) {
                    block.resize(block_count);
                }
                
                // std::cout << "block_count = " << block_count << std::endl;
                
                // the number of successors copied, and the total number of successors specified
                // in some copy
                int copied = 0, total = 0;
                
                for (i = 0; i < block_count; i++) {
                    block[i] = read_block() + (i == 0 ? 0 : 1);
                    total += block[i];
                    if (i % 2 == 0) {
                        copied += block[i];
                    }
                }
                if (block_count%2 == 0) {
                    copied += (outd[ref_index] - total);
                }
                extra_count = d - copied;
            }
            else {
                extra_count = d;
            }
            
            int interval_count = 0;
            if (extra_count > 0)
            {
                if (min_interval_length != 0 && (interval_count = bis.read_gamma()) != 0) 
                {
                    int prev = 0;
                    left.resize(interval_count);
                    len.resize(interval_count);
                    
                    // now read the intervals
                    left[0] = prev = nat2int(bis.read_gamma()) + x;
                    len[0] = bis.read_gamma() + min_interval_length;
                    
                    prev += len[0];
                    extra_count -= len[0];
                    
                    for (i=1; i < interval_count; i++) {
                        left[i] = prev = bis.read_gamma() + prev + 1;
                        len[i] = bis.read_gamma() + min_interval_length;
                        prev += len[i];
                        extra_count -= len[i];
                    }
                }
            }
            
            // allocate a sufficient buffer for the output
            if (arcs.size() < (unsigned)d) {
                arcs.resize(d);
                buffer1.resize(d);
                buffer2.resize(d);
            }
            
            int buf1_index = 0;
            int buf2_index = 0;
            
            // std::cout << "extra_count = " << extra_count << std::endl;
            // std::cout << "interval_count = " << interval_count << std::endl;
            // std::cout << "ref = " << ref << std::endl;
            
            // read the residuals into a buffer
            {
                int prev = -1;
                int residual_count = extra_count;
                while (residual_count > 0) {
                    residual_count--;
                    if (prev == -1) { buffer1[buf1_index++] = prev = x + nat2int(read_residual()); }
                    else { buffer1[buf1_index++] = prev = read_residual() + prev + 1; }
                }
                // std::cout << "residuals: buf1" << std::endl;
                // std::copy(buffer1.begin(),buffer1.begin()+buf1_index,std::ostream_iterator<int>(std::cout, "\n"));
            }
            // std::cout << "buf1_index = " << buf1_index << std::endl;
                
            if (interval_count == 0)
            {
                // don't do anything
            }
            else
            {
                // copy the extra interval data
                for (i = 0; i < interval_count; i++)
                {
                    int cur_left = left[i];
                    for (int j = 0; j < len[i]; j++) {
                        buffer2[buf2_index++] = cur_left + j;
                    }
                }
                
                // std::cout << "sequences: buf2" << std::endl;
                // std::copy(buffer2.begin(),buffer2.begin()+buf2_index,std::ostream_iterator<int>(std::cout, "\n"));
                
                if (extra_count > 0)
                {
                    std::merge(
                        buffer1.begin(), buffer1.begin()+buf1_index,
                        buffer2.begin(), buffer2.begin()+buf2_index,
                        arcs.begin()
                        );
                    buf1_index = buf1_index + buf2_index;
                    buf2_index = 0;           
                    std::copy(arcs.begin(), arcs.end(),
                        buffer1.begin());
                }
                else
                {
                    std::copy(buffer2.begin(), buffer2.begin()+buf2_index,
                        buffer1.begin());
                    buf1_index = buf2_index;
                    buf2_index = 0;
                }
            }
            
            if (ref <= 0)
            {
                // don't do anything except copy
                // the data to arcs
                if (interval_count == 0 || extra_count == 0) {
                    std::copy(buffer1.begin(), buffer1.end(),
                        arcs.begin()
                        );
                }
            }
            else
            {          
                // TODO clean this code up          
                // copy the information from the masked iterator
                
                int mask_index = 0;
                int len = 0;
                for (i=0; i < outd[ref_index]; )
                {
                    if (len <= 0)
                    {
                        if (block_count == mask_index) 
                        {
                            if (block_count % 2 == 0) {
                                len = outd[ref_index] - i;
                            }
                            else {
                                break;
                            }
                        }
                        else {
                            if (mask_index % 2 == 0) { len = block[mask_index++]; }
                            else { i += block[mask_index++]; continue; }
                        }
                        
                        // in the case that length is 0, we continue.
                        if (len == 0) { continue; }
                    }
                    buffer2[buf2_index++] = window[ref_index][i];
                    len--;
                    i++;
                }
                
                // std::cout << "masked" << std::endl;
                // std::copy(buffer2.begin(),buffer2.begin()+buf2_index,std::ostream_iterator<int>(std::cout, "\n"));
                
                std::merge(
                    buffer1.begin(), buffer1.begin()+buf1_index,
                    buffer2.begin(), buffer2.begin()+buf2_index,
                    arcs.begin() 
                    );
                buf1_index = buf1_index + buf2_index;
                buf2_index = 0;
                
            }

            // std::cout << "arcs" << std::endl;
            // std::copy(arcs.begin(),arcs.begin()+d,std::ostream_iterator<int>(std::cout, "\n"));
            assert (buf1_index == d);
            // std::cout << "end arcs" << std::endl;
        }
    } // namespace impl            
}// namespace yasmic


