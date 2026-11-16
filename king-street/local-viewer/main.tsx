import {createRoot} from 'react-dom/client';
import '../app/globals.css';

const route = location.pathname.replace(/\/$/, '');
const page = route === '/block-drive' ? import('../app/block-drive/page') :
  route === '/car-studio' ? import('../app/car-studio/page') : import('../app/king-pilot');
void page.then(({default: Page}) => createRoot(document.getElementById('root')!).render(<Page/>));
