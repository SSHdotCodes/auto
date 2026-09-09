import { test, expect } from "@playwright/test";

test("hero loads, assets stay local, and layout fits the viewport", async ({
  page,
}, testInfo) => {
  const errors = [],
    failures = [],
    external = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("response", (response) => {
    if (response.status() >= 400) failures.push(response.url());
  });
  page.on("request", (request) => {
    if (!request.url().startsWith("http://127.0.0.1:8542"))
      external.push(request.url());
  });
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText(
    "A little intelligence.",
  );
  await expect
    .poll(() =>
      page
        .locator(".hero-image img")
        .evaluate((img) => img.complete && img.naturalWidth > 800),
    )
    .toBe(true);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("hero.png") });
  for (const section of ["#benchmarks", "#in-context", "#install", "#models"]) {
    await page.locator(section).scrollIntoViewIfNeeded();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    ).toBe(true);
  }
  expect(errors).toEqual([]);
  expect(failures).toEqual([]);
  expect(external).toEqual([]);
});

test("chart controls preserve the metric, denominator, and zero-based axis", async ({
  page,
}) => {
  await page.goto("/#benchmarks");
  await page.locator('[data-metric="false_approve"]').click();
  await expect(page.locator("#metric-title")).toContainText("Lower is better");
  await expect(page.locator("#chart")).toHaveAttribute(
    "aria-label",
    /auto-0.4b-2 3.14 percent/,
  );
  await expect(page.locator(".chart-axis span").first()).toHaveText("0%");
  await expect(page.locator(".chart-axis span").last()).toHaveText("10%");
  await expect(page.locator("#metric-description")).toContainText("1,401");
  await page.locator('[data-metric="false_deny"]').click();
  await expect(page.locator("#chart")).toHaveAttribute(
    "aria-label",
    /auto-0.4b-2 3.31 percent/,
  );
  await page.locator('[data-metric="accuracy"]').click();
  await expect(page.locator("#chart")).toHaveAttribute(
    "aria-label",
    /auto-0.4b-2 96.77 percent/,
  );
});

test("every install combination and contextual scenario works", async ({
  page,
}) => {
  await page.goto("/#install");
  for (const agent of ["pi", "opencode", "hermes"]) {
    await page.locator(`[data-agent="${agent}"]`).click();
    for (const os of ["unix", "windows"]) {
      await page.locator(`[data-os="${os}"]`).click();
      await expect(page.locator("#install-command")).toContainText(
        os === "unix" ? `--agent ${agent}` : `-Agent ${agent}`,
      );
      await expect(page.locator("#install-command")).toContainText(
        os === "unix" ? "install.sh" : "install.ps1",
      );
    }
  }
  await page.locator("#copy-command").click();
  await expect(page.locator("#copy-status")).toContainText(
    /Copied|Command selected/,
  );
  await page.locator("#probe-select").selectOption("1");
  await expect(page.locator("#probe-decision")).toHaveText("Deny");
  await expect(page.locator("#probe-request")).toContainText(
    "Preserve all existing records",
  );
  await page.locator("#probe-select").selectOption("8");
  await expect(page.locator("#probe-decision")).toHaveText("Approve");
  await expect(page.locator("#probe-call")).toContainText(
    "mcp__gcal__list_events",
  );
});

test("origin serves installers, rejects private files, and reports readiness", async ({
  request,
}) => {
  for (const name of ["install.sh", "install.ps1", "install.py"])
    expect((await request.get("/" + name)).ok()).toBe(true);
  for (const name of ["serve.py", ".git/config", "assets/mountain.png"])
    expect((await request.get("/" + name)).status()).toBe(404);
  const result = await request.get("/healthz");
  expect(await result.json()).toMatchObject({ status: "ok", app: "auto" });
  expect(result.headers()["x-content-type-options"]).toBe("nosniff");
});
