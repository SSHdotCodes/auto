import {defineConfig, devices} from '@playwright/test';

export default defineConfig({
  testDir: './tests/site',
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', {open: 'never'}]],
  use: {baseURL: 'http://127.0.0.1:8542', screenshot: 'only-on-failure', trace: 'retain-on-failure'},
  projects: [
    {name: 'chromium-desktop', use: {...devices['Desktop Chrome']}},
    {name: 'firefox-desktop', use: {...devices['Desktop Firefox']}},
    {name: 'webkit-mobile', use: {...devices['iPhone 13']}},
    {name: 'small-mobile', use: {browserName: 'chromium', viewport: {width: 360, height: 800}}},
  ],
  webServer: {command: 'python site/serve.py --port 8542', url: 'http://127.0.0.1:8542/healthz', reuseExistingServer: !process.env.CI},
});
