import os
import time
import asyncio
import aiohttp
import aiofiles
import threading
import hashlib
from typing import Set, List, Tuple
from PIL import Image
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class NSFWImageDownloader:
    def __init__(self,
                 data_dir: str = RAW_DATA_DIR,
                 output_dir: str = NSFW_IMAGES_DIR,
                 max_images: int = UNDETERMINED_DATASET_IMAGES,
                 max_workers: int = 50,
                 chunk_size: int = 1024 * 1024,
                 retry_attempts: int = 3):

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_images = max_images
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.retry_attempts = retry_attempts
        self.download_count = 0
        self.failed_downloads: Set[str] = set()
        self.successful_downloads: Set[str] = set()
        self.download_lock = threading.Lock()
        self.counter_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.url_queue = asyncio.Queue()
        self.start_time = time.time()
        self.url_cache = set()

        # Progress bars
        self.main_pbar = None
        self.download_pbar = None

        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize session
        self.session = self._create_session()

        # Semaphore for concurrent downloads
        self.semaphore = asyncio.Semaphore(max_workers)

    @staticmethod
    def _create_session():
        """Create an optimized requests session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=100,
            pool_maxsize=100,
            pool_block=False
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @staticmethod
    def find_txt_files(directory):
        """Recursively find all .txt files in directory and subdirectories"""
        txt_files = [os.path.join(root, file)
                     for root, _, files in os.walk(directory)
                     for file in files if file.endswith(".txt")
                     ]

        return sorted(txt_files)

    @staticmethod
    async def verify_image_corruption(file_path: str) -> bool:
        """Verify image and attempt repair if corrupted. Delete if unfixable."""
        try:
            repair = ImageRepair(file_path)
            diagnosis = repair.diagnose()

            if diagnosis["is_valid_image"]:
                return True

            # If we get here, image is corrupted and couldn't be fixed
            if os.path.exists(file_path):
                os.remove(file_path)
                # logger.warning(f"Deleted corrupted image that couldn't be fixed: {file_path}")
            return False

        except Exception as e:
            # logger.error(f"Error during image verification/repair: {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)
                # logger.warning(f"Deleted image due to verification error: {file_path}")
            return False

    @staticmethod
    async def verify_image(image_data: bytes, file_path: str) -> bool:
        """Verify image integrity using multiple checks"""
        try:
            # Quick memory check with PIL
            try:
                img = Image.open(io.BytesIO(image_data))
                img.verify()
                img.close()

                # Reopen for size check
                img = Image.open(io.BytesIO(image_data))
                if img.size[0] < 10 or img.size[1] < 10:
                    return False

                # Check for corrupted or empty images
                if img.getbbox() is None:
                    return False

                img.close()
            except (SystemError, IOError):
                return False

            # Save temp file for thorough verification
            temp_path = f"{file_path}.temp"
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(image_data)

            # Use ImageRepair for thorough verification
            repair = ImageRepair(temp_path)
            diagnosis = repair.diagnose()

            if os.path.exists(temp_path):
                os.remove(temp_path)

            return diagnosis["is_valid_image"] and not diagnosis["issues"]
        except Exception as e:
            # logger.error(f"Image verification failed: {str(e)}")
            return False

    async def _download_with_aiohttp(self, url: str, filename: str, session: aiohttp.ClientSession) -> bool:
        """Optimized asynchronous download using aiohttp with chunked transfer"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    print(f"Failed to download {url}: HTTP status {response.status}")
                    return False
                
                # Check if content type is an image
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    print(f"Skipping {url}: Not an image (content-type: {content_type})")
                    return False
                
                # Get content length if available
                content_length = response.headers.get('content-length')
                if content_length:
                    content_length = int(content_length)
                    if content_length > 50 * 1024 * 1024:  # Skip files larger than 50MB
                        print(f"Skipping {url}: File too large ({content_length/1024/1024:.1f}MB)")
                        return False
                
                async with aiofiles.open(filename, 'wb') as f:
                    try:
                        async for chunk in response.content.iter_chunked(self.chunk_size):
                            if self.stop_event.is_set():
                                return False
                            await f.write(chunk)
                        return True
                    except Exception as e:
                        print(f"Error while downloading {url}: {str(e)}")
                        if os.path.exists(filename):
                            os.remove(filename)
                        return False
                        
        except aiohttp.ClientError as e:
            print(f"Network error for {url}: {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error downloading {url}: {str(e)}")
            return False

    async def _download_image(self, url: str, session: aiohttp.ClientSession, image_number: int) -> Tuple[bool, str]:
        """Download image with optimized async operations"""
        if self.stop_event.is_set() or self.download_count >= self.max_images:
            return False, ""

        async with self.semaphore:
            filename = os.path.join(self.output_dir, f"nsfw_{image_number}.jpg")
            
            # Skip if file exists and is valid
            if os.path.exists(filename):
                try:
                    with Image.open(filename) as img:
                        img.verify()
                    return True, filename
                except Exception:
                    os.remove(filename)

            for attempt in range(self.retry_attempts):
                try:
                    # Download with optimized async client
                    if await self._download_with_aiohttp(url, filename, session):
                        # Verify downloaded image
                        try:
                            with Image.open(filename) as img:
                                img.verify()
                            return True, filename
                        except Exception as e:
                            print(f"Invalid image downloaded from {url}: {str(e)}")
                            if os.path.exists(filename):
                                os.remove(filename)
                    
                    if attempt < self.retry_attempts - 1:
                        print(f"Retrying download for {url} (attempt {attempt + 2}/{self.retry_attempts})")
                        await asyncio.sleep(1)

                except Exception as e:
                    print(f"Error during download attempt {attempt + 1} for {url}: {str(e)}")
                    if os.path.exists(filename):
                        os.remove(filename)
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(1)

            return False, ""

    def _print_tqdm(self):
        """Initialize progress bars for download tracking"""
        # Clear any existing progress bars
        if self.main_pbar:
            self.main_pbar.close()
        if self.download_pbar:
            self.download_pbar.close()

        progress_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        self.main_pbar = tqdm(
            total=self.max_images,
            desc="Total Progress",
            unit="img",
            position=0,
            bar_format=progress_format,
            file=tqdm_out,
            dynamic_ncols=True
        )
        self.download_pbar = tqdm(
            total=self.max_images,
            desc="Downloaded   ",
            unit="img",
            position=1,
            bar_format=progress_format,
            file=tqdm_out,
            dynamic_ncols=True
        )

    async def _verify_and_repair_image(self, filename: str) -> bool:
        """Verify and repair downloaded image"""
        try:
            if await self.verify_image_corruption(filename):
                return True

            repair = ImageRepair(filename)
            diagnosis = repair.diagnose()

            if not diagnosis["is_valid_image"]:
                if os.path.exists(filename):
                    os.remove(filename)
                return False

            return True

        except Exception as e:
            # logger.error(f"Verification/repair failed for image {filename}: {str(e)}")
            if os.path.exists(filename):
                os.remove(filename)
            return False

    async def download_image(self, url: str, session: aiohttp.ClientSession, image_number: int):
        """Download and verify a single image with optimized async operations"""
        try:
            filename = os.path.join(self.output_dir, f"nsfw_{image_number}.jpg")
            
            # Download the image
            success, filename = await self._download_image(url, session, image_number)
            
            # Verify and repair if needed
            if success:
                with self.counter_lock:  # Add atomic counter operation
                    self.download_count += 1
                    if self.download_pbar:
                        self.download_pbar.update(1)
                self.successful_downloads.add(url)
                
                if self.max_images != UNDETERMINED_DATASET_IMAGES:
                    with self.counter_lock:  # Add atomic check for max images
                        if self.download_count >= self.max_images:
                            self.stop_event.set()
            else:
                self.failed_downloads.add(url)
                
        except Exception as e:
            self.failed_downloads.add(url)
            # logger.error(f"Error downloading {url}: {str(e)}")
            
    @staticmethod
    async def _load_urls_from_file(file_path: str) -> List[str]:
        """Optimized async file reading for a single file"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
                return [url.strip() for url in content.split('\n') if url.strip()]
        except Exception as e:
            # logger.error(f"Error reading file {file_path}: {str(e)}")
            return []

    async def _process_urls_batch(self, urls_batch: List[str]):
        """Process URLs in batches for efficiency"""
        if self.download_count >= self.max_images:
            return

        unique_urls = set(urls_batch) - self.url_cache
        self.url_cache.update(unique_urls)

        for url in unique_urls:
            if self.download_count >= self.max_images:
                break
            await self.url_queue.put(url)
    async def _load_urls_from_all_files(self, data_dir: str) -> List[str]:
        """Load URLs from all text files in directory with optimized async file reading"""
        try:
            txt_files = self.find_txt_files(data_dir)
            if not txt_files:
                # logger.error("No .txt files found!")
                return []

            total_urls = 0
            total_files = 0
            urls_batch = []
            tasks = []

            for file_path in txt_files:
                tasks.append(self._load_urls_from_file(file_path))
                total_files += 1
                if total_files % 10 == 0:
                    # logger.debug(f"Processing {total_files}/{len(txt_files)} files")
                    pass

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    # logger.error(f"Error reading file: {str(result)}")
                    pass
                else:
                    urls_batch.extend(result)

            return urls_batch

        except Exception as e:
            # logger.error(f"Error loading URLs: {str(e)}")
            pass

    async def _process_urls(self, urls_batch: List[str]):
        """Process URLs in batches for efficiency"""
        if self.download_count >= self.max_images:
            return

        unique_urls = set(urls_batch) - self.url_cache
        self.url_cache.update(unique_urls)

        for url in unique_urls:
            if self.download_count >= self.max_images:
                break
            await self.url_queue.put(url)

    async def load_urls_from_files(self):
        """Load URLs from text files with optimized async file reading"""
        try:
            data_dir = os.path.join(self.data_dir, 'raw_data')
            if not os.path.exists(data_dir):
                # logger.error(f"Data directory not found: {data_dir}")
                return

            urls_batch = await self._load_urls_from_all_files(data_dir)
            await self._process_urls(urls_batch)

        except Exception as e:
            # logger.error(f"Error loading URLs: {str(e)}")
            pass

    async def download_worker(self, worker_id: int, session: aiohttp.ClientSession):
        """Optimized worker for processing URLs from the queue"""
        while not self.stop_event.is_set():
            try:
                if self.download_count >= self.max_images:
                    break

                # Add timeout to queue.get()
                try:
                    url = await asyncio.wait_for(self.url_queue.get(), timeout=10)
                except asyncio.TimeoutError:
                    # Only stop if we haven't reached our target
                    if self.download_count < self.max_images:
                        # Put more URLs into the queue if available
                        await self.load_urls_from_files()
                        if self.url_queue.empty():
                            print(f"Worker {worker_id}: No more URLs available. Current count: {self.download_count}/{self.max_images}")
                            break
                    else:
                        break

                current_count = self.download_count + 1

                # Skip if URL is already processed
                url_hash = hashlib.md5(url.encode()).hexdigest()
                if url_hash in self.successful_downloads:
                    self.url_queue.task_done()
                    continue

                success = await self.download_image(url, session, current_count)

                if success:
                    self.successful_downloads.add(url_hash)
                else:
                    self.failed_downloads.add(url)
                    # If download failed, put a new URL in the queue if we haven't reached our target
                    if self.download_count < self.max_images:
                        await self.load_urls_from_files()

                self.url_queue.task_done()

                # Update progress bars
                if self.main_pbar:
                    self.main_pbar.update(1)

                # Rate limiting
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker {worker_id} error: {str(e)}")
                if 'url' in locals():
                    self.url_queue.task_done()

    async def run(self):
        """Main execution function with optimized async operations"""
        try:
            # Initialize progress bars
            self._print_tqdm()

            # Optimized connection pooling
            connector = aiohttp.TCPConnector(
                limit=self.max_workers,
                force_close=True,
                enable_cleanup_closed=True
            )

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'image/*',
                    'Connection': 'keep-alive'
                }
            ) as session:
                # Load URLs first
                print("Loading URLs from files...")
                await self.load_urls_from_files()
                
                if self.url_queue.empty():
                    print("No URLs found to process. Check your data directory.")
                    return

                print(f"Starting {self.max_workers} download workers...")
                # Create and run workers
                workers = []
                for i in range(self.max_workers):
                    worker = asyncio.create_task(self.download_worker(i, session))
                    workers.append(worker)

                # Wait for all workers to complete or until we reach max_images
                while not self.stop_event.is_set():
                    if self.download_count >= self.max_images:
                        print(f"\nReached target of {self.max_images} images. Stopping...")
                        self.stop_event.set()
                        break
                        
                    if all(worker.done() for worker in workers):
                        if self.download_count < self.max_images:
                            print(f"\nAll workers finished but only downloaded {self.download_count}/{self.max_images} images.")
                            print("This might be due to insufficient valid URLs or too many failed downloads.")
                        break
                        
                    await asyncio.sleep(1)

                # Cancel remaining workers
                for worker in workers:
                    if not worker.done():
                        worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

        except Exception as e:
            print(f"Error in main execution: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Close progress bars
            if self.main_pbar:
                self.main_pbar.close()
            if self.download_pbar:
                self.download_pbar.close()
            self.print_summary()

    def print_summary(self):
        """Print detailed download summary to log file only"""
        elapsed_time = time.time() - self.start_time
        print(f"\nDownload Summary:")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Successfully downloaded: {len(self.successful_downloads)} images")
        print(f"Failed downloads: {len(self.failed_downloads)} URLs")
        print(f"Average download rate: {len(self.successful_downloads)/elapsed_time:.2f} images/second")
